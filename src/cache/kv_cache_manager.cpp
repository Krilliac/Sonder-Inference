// Sonder Inference: KV-cache module: logical block/page manager.
#include "sonder/inference/cache/kv_cache_manager.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace sonder::inference::cache {
namespace {

constexpr std::uint64_t mix64(std::uint64_t x) noexcept {
    // splitmix64 finaliser
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::string seq_str(SequenceId id) { return std::to_string(id); }

}  // namespace

KvCacheManager::KvCacheManager(KvCacheConfig config, std::unique_ptr<EvictionPolicy> policy)
    : config_(config), policy_(std::move(policy)) {
    if (config_.block_size_tokens == 0) {
        throw std::invalid_argument("KvCacheConfig::block_size_tokens must be > 0");
    }
    if (!(config_.high_watermark > 0.0 && config_.high_watermark <= 1.0) ||
        !(config_.critical_watermark > 0.0 && config_.critical_watermark <= 1.0) ||
        config_.high_watermark > config_.critical_watermark) {
        throw std::invalid_argument("KvCacheConfig watermarks must satisfy 0 < high <= critical <= 1");
    }
    if (!policy_) policy_ = std::make_unique<PriorityAwareEvictionPolicy>();
    blocks_.resize(config_.num_blocks);
    free_list_.reserve(config_.num_blocks);
    for (std::uint32_t i = config_.num_blocks; i > 0; --i) free_list_.push_back(i - 1);
}

// --- hashing / index ---------------------------------------------------------

std::uint64_t KvCacheManager::chain_hash(std::uint64_t parent, const CacheFingerprint& fp,
                                         std::span<const TokenId> tokens,
                                         ModelArchitecture architecture) const noexcept {
    std::uint64_t h = mix64(parent ^ mix64(fp.value ^ 0x5bd1e9955bd1e995ULL));
    if (architecture != ModelArchitecture::attention_only) {
        h = mix64(h ^ mix64(static_cast<std::uint8_t>(architecture)));
    }
    for (const TokenId t : tokens) {
        h = mix64(h ^ static_cast<std::uint32_t>(t));
    }
    return h;
}

std::uint64_t KvCacheManager::prev_hash(const Sequence& seq, std::size_t block_index) const noexcept {
    if (block_index == 0) return 0;
    return blocks_[seq.blocks[block_index - 1]].hash;
}

std::optional<BlockId> KvCacheManager::lookup(std::uint64_t hash, std::uint64_t parent,
                                              const CacheFingerprint& fp,
                                              std::span<const TokenId> tokens,
                                              ModelArchitecture architecture) const {
    const auto it = prefix_index_.find(hash);
    if (it == prefix_index_.end()) return std::nullopt;
    const Block& blk = blocks_[it->second];
    // Correctness beats hit rate: verify everything we can, not just the hash.
    if (blk.parent_hash != parent || !(blk.fingerprint == fp) || blk.architecture != architecture ||
        blk.tokens.size() != tokens.size() ||
        !std::equal(tokens.begin(), tokens.end(), blk.tokens.begin())) {
        return std::nullopt;
    }
    return it->second;
}

void KvCacheManager::unregister(BlockId b) {
    Block& blk = blocks_[b];
    if (!blk.registered) return;
    const auto it = prefix_index_.find(blk.hash);
    if (it != prefix_index_.end() && it->second == b) prefix_index_.erase(it);
    blk.registered = false;
}

void KvCacheManager::finalize_block(Sequence& seq, std::size_t block_index) {
    const BlockId b = seq.blocks[block_index];
    Block& blk = blocks_[b];
    blk.parent_hash = prev_hash(seq, block_index);
    blk.fingerprint = seq.options.fingerprint;
    blk.architecture = seq.options.architecture;
    blk.hash = chain_hash(blk.parent_hash, blk.fingerprint, blk.tokens, blk.architecture);
    blk.has_hash = true;
    blk.depth = static_cast<std::uint32_t>(block_index);
    if (config_.enable_prefix_caching && !prefix_index_.contains(blk.hash)) {
        prefix_index_.emplace(blk.hash, b);
        blk.registered = true;
    }
}

// --- block pool --------------------------------------------------------------

void KvCacheManager::reset_block(BlockId b) {
    remove_checkpoints_for_block(b);
    Block& blk = blocks_[b];
    blk.tokens.clear();
    blk.has_hash = false;
    blk.registered = false;
    blk.evictable = false;
    blk.hash = 0;
    blk.parent_hash = 0;
    blk.depth = 0;
    blk.priority = 0;
    blk.fingerprint = {};
    blk.architecture = ModelArchitecture::attention_only;
}

BlockId KvCacheManager::allocate_block(SequenceId owner) {
    BlockId b = kInvalidBlock;
    if (!free_list_.empty()) {
        b = free_list_.back();
        free_list_.pop_back();
    } else if (const auto victim = policy_->pop_victim()) {
        b = *victim;
        blocks_[b].evictable = false;
        unregister(b);
        reset_block(b);
        ++counters_.evictions;
        emit({CacheEvent::Kind::evicted, 0, b, kInvalidBlock, PressureLevel::normal});
    } else {
        return kInvalidBlock;
    }
    blocks_[b].tokens.reserve(config_.block_size_tokens);
    ++counters_.blocks_allocated;
    emit({CacheEvent::Kind::allocated, owner, b, kInvalidBlock, PressureLevel::normal});
    return b;
}

void KvCacheManager::acquire(BlockId b, const Sequence& seq) {
    Block& blk = blocks_[b];
    if (blk.evictable) {
        policy_->erase(b);
        blk.evictable = false;
    }
    ++blk.ref_count;
    blk.priority = std::max(blk.priority, seq.options.priority);
    blk.last_use = ++clock_;
}

void KvCacheManager::release(BlockId b, Priority priority) {
    Block& blk = blocks_[b];
    --blk.ref_count;
    blk.priority = std::max(blk.priority, priority);
    blk.last_use = ++clock_;
    if (blk.ref_count != 0) return;
    if (blk.registered && config_.enable_prefix_caching) {
        blk.evictable = true;
        EvictionCandidate c;
        c.block = b;
        c.last_use = blk.last_use;
        c.priority = blk.priority;
        c.depth = blk.depth;
        c.recompute_tokens = static_cast<std::uint64_t>(blk.depth + 1) * config_.block_size_tokens;
        policy_->insert(c);
    } else {
        unregister(b);
        reset_block(b);
        free_list_.push_back(b);
    }
}

void KvCacheManager::ensure_tail_writable(SequenceId id, Sequence& seq, AppendResult& result) {
    const std::size_t k = seq.num_tokens % config_.block_size_tokens;
    const BlockId b = seq.blocks.back();
    Block& blk = blocks_[b];
    if (blk.ref_count > 1 || (seq.options.architecture != ModelArchitecture::attention_only &&
                              checkpoint_uses_block(b))) {
        const BlockId nb = allocate_block(id);  // availability pre-checked by caller
        Block& nblk = blocks_[nb];
        nblk.tokens.assign(blk.tokens.begin(), blk.tokens.begin() + static_cast<std::ptrdiff_t>(k));
        nblk.fingerprint = seq.options.fingerprint;
        acquire(nb, seq);
        // A non-indexed tail will be released below. Preserve checkpoints whose
        // entire prefix survives the copy by moving their lineage reference.
        if (!blk.registered && blk.ref_count == 1) {
            for (auto& cp : checkpoints_) {
                for (std::size_t index = 0; index < cp.blocks.size(); ++index) {
                    if (cp.blocks[index] == b &&
                        cp.state.position - index * config_.block_size_tokens <= k) cp.blocks[index] = nb;
                }
            }
        }
        release(b, seq.options.priority);
        seq.blocks.back() = nb;
        pending_copies_.push_back({b, nb, static_cast<std::uint32_t>(k)});
        ++counters_.copy_on_write;
        ++result.copy_on_write;
        ++result.blocks_allocated;
        emit({CacheEvent::Kind::copy_on_write, id, nb, b, PressureLevel::normal});
    } else {
        // Exclusively owned: mutate in place, but it can no longer serve as a
        // cached prefix block (e.g. after truncate into a full block).
        unregister(b);
        blk.has_hash = false;
        blk.tokens.resize(k);
    }
}

// --- sequence lifecycle --------------------------------------------------------

Status KvCacheManager::add_sequence(SequenceId id, const SequenceOptions& options) {
    if (sequences_.contains(id)) {
        return Status(ErrorCode::invalid_state, "sequence " + seq_str(id) + " already exists");
    }
    // Owner IDs may be recycled, but old backend state must not be attributed
    // to the new owner incarnation.
    for (std::size_t i = checkpoints_.size(); i > 0; --i) {
        if (checkpoints_[i - 1].state.owner == id) erase_checkpoint(i - 1);
    }
    Sequence seq;
    seq.options = options;
    sequences_.emplace(id, std::move(seq));
    return Status::success();
}

Status KvCacheManager::save_checkpoint(SequenceId id, std::size_t position,
                                       std::uint64_t state_size) {
    const auto it = sequences_.find(id);
    if (it == sequences_.end()) return Status(ErrorCode::not_found, "unknown sequence " + seq_str(id));
    const Sequence& seq = it->second;
    if (seq.options.architecture == ModelArchitecture::attention_only || position == 0 ||
        position > seq.num_tokens) {
        return Status(ErrorCode::invalid_argument, "invalid recurrent checkpoint position or architecture");
    }
    if (state_size == 0 || config_.max_checkpoints == 0 || state_size > config_.max_checkpoint_bytes) {
        return Status(ErrorCode::invalid_argument, "checkpoint exceeds configured capacity");
    }
    Checkpoint saved;
    saved.state = {position, state_size, id};
    saved.fingerprint = seq.options.fingerprint;
    saved.architecture = seq.options.architecture;
    saved.blocks.assign(seq.blocks.begin(), seq.blocks.begin() +
                        static_cast<std::ptrdiff_t>(blocks_for_tokens(position)));
    // Validate before mutating accounting. Subtraction keeps UINT64_MAX budgets safe.
    for (std::size_t i = checkpoints_.size(); i > 0; --i) {
        if (checkpoints_[i - 1].state.owner == id && checkpoints_[i - 1].state.position == position) {
            erase_checkpoint(i - 1);
        }
    }
    std::uint64_t total = 0;
    for (const auto& cp : checkpoints_) total += cp.state.state_size;
    while (!checkpoints_.empty() &&
           (total > config_.max_checkpoint_bytes - state_size || checkpoints_.size() >= config_.max_checkpoints)) {
        const auto victim = std::min_element(checkpoints_.begin(), checkpoints_.end(),
                                            [](const Checkpoint& a, const Checkpoint& b) {
                                                return a.last_use < b.last_use;
                                            });
        total -= victim->state.state_size;
        erase_checkpoint(static_cast<std::size_t>(victim - checkpoints_.begin()));
    }
    saved.last_use = ++clock_;
    checkpoints_.push_back(std::move(saved));
    return Status::success();
}

Status KvCacheManager::remove_checkpoint(SequenceId id, std::size_t position) {
    for (std::size_t i = 0; i < checkpoints_.size(); ++i) {
        if (checkpoints_[i].state.owner == id && checkpoints_[i].state.position == position) {
            erase_checkpoint(i);
            return Status::success();
        }
    }
    return Status(ErrorCode::not_found, "checkpoint not found");
}

std::vector<TokenId> KvCacheManager::sequence_tokens(const Sequence& seq) const {
    std::vector<TokenId> tokens;
    tokens.reserve(seq.num_tokens);
    for (const BlockId b : seq.blocks) {
        const auto& block = blocks_[b].tokens;
        const auto take = std::min(seq.num_tokens - tokens.size(), block.size());
        tokens.insert(tokens.end(), block.begin(), block.begin() + static_cast<std::ptrdiff_t>(take));
    }
    return tokens;
}

std::optional<RecurrentStateCheckpoint> KvCacheManager::checkpoint_at(SequenceId id,
                                                                     std::size_t position) const {
    const auto it = sequences_.find(id);
    if (it == sequences_.end() || position > it->second.num_tokens) return std::nullopt;
    const auto& seq = it->second;
    const auto tokens = sequence_tokens(seq);
    const auto cp = find_checkpoint(seq.options.fingerprint, seq.options.architecture, tokens, position);
    if (!cp || checkpoints_[*cp].state.position != position) return std::nullopt;
    return checkpoints_[*cp].state;
}

Status KvCacheManager::fork_sequence(SequenceId parent, SequenceId child,
                                     std::optional<Priority> child_priority) {
    const auto pit = sequences_.find(parent);
    if (pit == sequences_.end()) {
        return Status(ErrorCode::not_found, "unknown parent sequence " + seq_str(parent));
    }
    if (sequences_.contains(child)) {
        return Status(ErrorCode::invalid_state, "sequence " + seq_str(child) + " already exists");
    }
    if (pit->second.options.architecture != ModelArchitecture::attention_only && pit->second.num_tokens != 0) {
        const auto state = checkpoint_at(parent, pit->second.num_tokens);
        if (!state || state->owner == child) {
            return Status(ErrorCode::invalid_state, "recurrent fork requires an exact saved checkpoint");
        }
    }
    // Discard state left by an earlier incarnation of the child ID.
    for (std::size_t i = checkpoints_.size(); i > 0; --i) {
        if (checkpoints_[i - 1].state.owner == child) erase_checkpoint(i - 1);
    }
    Sequence copy = pit->second;
    if (child_priority) copy.options.priority = *child_priority;
    for (const BlockId b : copy.blocks) acquire(b, copy);
    sequences_.emplace(child, std::move(copy));
    update_pressure();
    return Status::success();
}

Status KvCacheManager::append_tokens(SequenceId id, std::span<const TokenId> tokens,
                                     AppendResult* out) {
    AppendResult result;
    const auto it = sequences_.find(id);
    if (it == sequences_.end()) {
        return Status(ErrorCode::not_found, "unknown sequence " + seq_str(id));
    }
    Sequence& seq = it->second;
    const std::size_t n = tokens.size();
    if (n == 0) {
        if (out) *out = result;
        return Status::success();
    }
    const std::size_t need = blocks_needed(id, n);
    if (need > available_blocks()) {
        ++counters_.allocation_failures;
        update_pressure();
        return Status(ErrorCode::unavailable, "need " + std::to_string(need) + " blocks, " +
                                                  std::to_string(available_blocks()) + " available");
    }

    const std::size_t bs = config_.block_size_tokens;
    std::size_t recurrent_limit = 0;
    std::optional<RecurrentStateCheckpoint> selected;
    // No checkpoint scan or growing prefix copy on attention/decode paths.
    if (seq.options.architecture != ModelArchitecture::attention_only &&
        config_.enable_prefix_caching && seq.prefix_streak && n >= bs) {
        auto combined = sequence_tokens(seq);
        combined.insert(combined.end(), tokens.begin(), tokens.end());
        const auto raw = match_blocks(seq.options.fingerprint, combined, seq.options.architecture);
        const auto cp = find_checkpoint(seq.options.fingerprint, seq.options.architecture, combined, raw);
        if (cp && checkpoints_[*cp].state.position > seq.num_tokens) {
            recurrent_limit = checkpoints_[*cp].state.position;
            selected = checkpoints_[*cp].state;
            // Restoring inside a block may need an additional COW block to
            // preserve the saved lineage. Fall back to recompute if the
            // worst-case reservation has no room for it.
            if (recurrent_limit % bs != 0 && need == available_blocks()) {
                recurrent_limit = 0;
                selected.reset();
            } else {
                checkpoints_[*cp].last_use = ++clock_;
            }
        }
        const auto usable = std::max(seq.num_tokens, recurrent_limit);
        result.tokens_checkpoint_limited = raw > usable ? raw - usable : 0;
        counters_.checkpoint_limited_tokens += result.tokens_checkpoint_limited;
    }
    std::size_t i = 0;
    while (i < n) {
        const std::size_t k = seq.num_tokens % bs;
        if (k != 0) {
            ensure_tail_writable(id, seq, result);
            Block& blk = blocks_[seq.blocks.back()];
            const std::size_t take = std::min(bs - k, n - i);
            blk.tokens.insert(blk.tokens.end(), tokens.begin() + static_cast<std::ptrdiff_t>(i),
                              tokens.begin() + static_cast<std::ptrdiff_t>(i + take));
            blk.last_use = ++clock_;
            seq.num_tokens += take;
            i += take;
            seq.prefix_streak = false;
            if (k + take == bs) finalize_block(seq, seq.blocks.size() - 1);
            continue;
        }

        const std::size_t remaining = n - i;
        if (config_.enable_prefix_caching && seq.prefix_streak && remaining >= bs) {
            const auto chunk = tokens.subspan(i, bs);
            const std::uint64_t parent = prev_hash(seq, seq.blocks.size());
            const std::uint64_t h = chain_hash(parent, seq.options.fingerprint, chunk, seq.options.architecture);
            if (seq.options.architecture == ModelArchitecture::attention_only ||
                recurrent_limit > seq.num_tokens) {
                if (const auto hit = lookup(h, parent, seq.options.fingerprint, chunk,
                                            seq.options.architecture)) {
                    const auto take = seq.options.architecture == ModelArchitecture::attention_only
                                          ? bs : std::min(bs, recurrent_limit - seq.num_tokens);
                    acquire(*hit, seq);
                    seq.blocks.push_back(*hit);
                    seq.num_tokens += take;
                    seq.cached_prefix += take;
                    i += take;
                    ++result.blocks_reused;
                    result.tokens_reused += take;
                    ++counters_.prefix_hit_blocks;
                    counters_.avoided_prefill_tokens += take;
                    emit({CacheEvent::Kind::reused, id, *hit, kInvalidBlock, PressureLevel::normal});
                    continue;
                }
            }
            ++counters_.prefix_miss_lookups;
        }
        seq.prefix_streak = false;

        const BlockId nb = allocate_block(id);  // pre-checked; cannot fail
        acquire(nb, seq);
        seq.blocks.push_back(nb);
        const std::size_t take = std::min(bs, remaining);
        Block& blk = blocks_[nb];
        blk.fingerprint = seq.options.fingerprint;
        blk.tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(i),
                          tokens.begin() + static_cast<std::ptrdiff_t>(i + take));
        seq.num_tokens += take;
        i += take;
        ++result.blocks_allocated;
        if (take == bs) finalize_block(seq, seq.blocks.size() - 1);
    }
    if (selected && result.tokens_reused > 0) {
        result.checkpoint = selected;
        counters_.checkpoint_reused_tokens += result.tokens_reused;
    }
    result.tokens_appended = n;
    if (out) *out = result;
    update_pressure();
    return Status::success();
}

Status KvCacheManager::truncate(SequenceId id, std::size_t new_length) {
    const auto it = sequences_.find(id);
    if (it == sequences_.end()) {
        return Status(ErrorCode::not_found, "unknown sequence " + seq_str(id));
    }
    Sequence& seq = it->second;
    if (new_length > seq.num_tokens) {
        return Status(ErrorCode::invalid_argument, "truncate beyond sequence length");
    }
    if (new_length == seq.num_tokens) return Status::success();
    if (seq.options.architecture != ModelArchitecture::attention_only && new_length != 0 &&
        !checkpoint_at(id, new_length)) {
        return Status(ErrorCode::invalid_argument, "recurrent truncate requires an exact saved checkpoint");
    }
    const std::size_t keep = blocks_for_tokens(new_length);
    while (seq.blocks.size() > keep) {
        release(seq.blocks.back(), seq.options.priority);
        seq.blocks.pop_back();
    }
    for (std::size_t i = checkpoints_.size(); i > 0; --i) {
        if (checkpoints_[i - 1].state.owner == id && checkpoints_[i - 1].state.position > new_length) {
            erase_checkpoint(i - 1);
        }
    }
    seq.num_tokens = new_length;
    seq.cached_prefix = std::min(seq.cached_prefix, new_length);
    seq.prefix_streak = seq.cached_prefix == seq.num_tokens && (seq.num_tokens % config_.block_size_tokens) == 0;
    update_pressure();
    return Status::success();
}

Status KvCacheManager::free_sequence(SequenceId id) {
    const auto it = sequences_.find(id);
    if (it == sequences_.end()) {
        return Status(ErrorCode::not_found, "unknown sequence " + seq_str(id));
    }
    Sequence& seq = it->second;
    // Release tail-first so roots of a prefix chain end up most recently used.
    for (auto b = seq.blocks.rbegin(); b != seq.blocks.rend(); ++b) release(*b, seq.options.priority);
    sequences_.erase(it);
    update_pressure();
    return Status::success();
}

Status KvCacheManager::set_priority(SequenceId id, Priority priority) {
    const auto it = sequences_.find(id);
    if (it == sequences_.end()) {
        return Status(ErrorCode::not_found, "unknown sequence " + seq_str(id));
    }
    it->second.options.priority = priority;
    for (const BlockId b : it->second.blocks) blocks_[b].priority = std::max(blocks_[b].priority, priority);
    return Status::success();
}

// --- queries -------------------------------------------------------------------

bool KvCacheManager::has_sequence(SequenceId id) const noexcept { return sequences_.contains(id); }

std::size_t KvCacheManager::num_tokens(SequenceId id) const noexcept {
    const auto it = sequences_.find(id);
    return it == sequences_.end() ? 0 : it->second.num_tokens;
}

std::size_t KvCacheManager::cached_prefix_tokens(SequenceId id) const noexcept {
    const auto it = sequences_.find(id);
    return it == sequences_.end() ? 0 : it->second.cached_prefix;
}

std::span<const BlockId> KvCacheManager::block_table(SequenceId id) const noexcept {
    const auto it = sequences_.find(id);
    if (it == sequences_.end()) return {};
    return it->second.blocks;
}

std::uint32_t KvCacheManager::ref_count(BlockId block) const noexcept {
    return block < blocks_.size() ? blocks_[block].ref_count : 0;
}

std::span<const TokenId> KvCacheManager::block_tokens(BlockId block) const noexcept {
    if (block >= blocks_.size()) return {};
    return blocks_[block].tokens;
}

std::size_t KvCacheManager::match_blocks(const CacheFingerprint& fingerprint,
                                         std::span<const TokenId> tokens, ModelArchitecture architecture) const {
    if (!config_.enable_prefix_caching) return 0;
    const std::size_t bs = config_.block_size_tokens;
    std::uint64_t parent = 0;
    std::size_t matched = 0;
    while (tokens.size() - matched >= bs) {
        const auto chunk = tokens.subspan(matched, bs);
        const auto h = chain_hash(parent, fingerprint, chunk, architecture);
        if (!lookup(h, parent, fingerprint, chunk, architecture)) break;
        parent = h;
        matched += bs;
    }
    return matched;
}

std::size_t KvCacheManager::match_prefix(const CacheFingerprint& fingerprint,
                                         std::span<const TokenId> tokens, ModelArchitecture architecture) const {
    const auto raw = match_blocks(fingerprint, tokens, architecture);
    if (architecture == ModelArchitecture::attention_only) return raw;
    const auto cp = find_checkpoint(fingerprint, architecture, tokens, raw);
    return cp ? checkpoints_[*cp].state.position : 0;
}

std::size_t KvCacheManager::blocks_for_tokens(std::size_t n) const noexcept {
    const std::size_t bs = config_.block_size_tokens;
    return (n + bs - 1) / bs;
}

std::size_t KvCacheManager::blocks_needed(SequenceId id, std::size_t n) const noexcept {
    const auto it = sequences_.find(id);
    if (it == sequences_.end()) return blocks_for_tokens(n);
    if (n == 0) return 0;
    const Sequence& seq = it->second;
    const std::size_t bs = config_.block_size_tokens;
    const std::size_t k = seq.num_tokens % bs;
    std::size_t need = 0;
    std::size_t room = 0;
    if (k != 0) {
        room = bs - k;
        if (blocks_[seq.blocks.back()].ref_count > 1 ||
            (seq.options.architecture != ModelArchitecture::attention_only &&
             checkpoint_uses_block(seq.blocks.back()))) need += 1;  // copy-on-write
    }
    const std::size_t rest = n > room ? n - room : 0;
    return need + blocks_for_tokens(rest);
}

std::size_t KvCacheManager::available_blocks() const noexcept {
    return free_list_.size() + policy_->size();
}

bool KvCacheManager::can_append(SequenceId id, std::size_t n) const noexcept {
    return has_sequence(id) && blocks_needed(id, n) <= available_blocks();
}

PressureLevel KvCacheManager::pressure() const noexcept {
    if (blocks_.empty()) return PressureLevel::normal;
    const std::size_t pinned = blocks_.size() - available_blocks();
    const double util = static_cast<double>(pinned) / static_cast<double>(blocks_.size());
    if (util >= config_.critical_watermark) return PressureLevel::critical;
    if (util >= config_.high_watermark) return PressureLevel::high;
    return PressureLevel::normal;
}

KvCacheStats KvCacheManager::stats() const {
    KvCacheStats s = counters_;
    s.total_blocks = blocks_.size();
    s.free_blocks = free_list_.size();
    s.cached_blocks = policy_->size();
    s.pinned_blocks = s.total_blocks - s.free_blocks - s.cached_blocks;
    s.shared_blocks = static_cast<std::size_t>(
        std::count_if(blocks_.begin(), blocks_.end(), [](const Block& b) { return b.ref_count > 1; }));
    s.sequences = sequences_.size();
    s.block_size_tokens = config_.block_size_tokens;
    s.bytes_per_block = config_.bytes_per_token * config_.block_size_tokens;
    s.pinned_bytes = s.bytes_per_block * s.pinned_blocks;
    s.cached_bytes = s.bytes_per_block * s.cached_blocks;
    s.checkpoints = checkpoints_.size();
    for (const auto& cp : checkpoints_) s.checkpoint_bytes += cp.state.state_size;
    s.utilization = s.total_blocks == 0 ? 0.0
                                        : static_cast<double>(s.pinned_blocks) / static_cast<double>(s.total_blocks);
    s.pressure = pressure();
    return s;
}

std::vector<BlockCopy> KvCacheManager::take_pending_copies() {
    std::vector<BlockCopy> out;
    out.swap(pending_copies_);
    return out;
}

std::size_t KvCacheManager::clear_cached() {
    std::size_t n = 0;
    while (const auto victim = policy_->pop_victim()) {
        const BlockId b = *victim;
        blocks_[b].evictable = false;
        unregister(b);
        reset_block(b);
        free_list_.push_back(b);
        ++counters_.evictions;
        emit({CacheEvent::Kind::evicted, 0, b, kInvalidBlock, PressureLevel::normal});
        ++n;
    }
    while (!checkpoints_.empty()) erase_checkpoint(checkpoints_.size() - 1);
    update_pressure();
    return n;
}

void KvCacheManager::emit(const CacheEvent& event) const {
    if (listener_) listener_(event);
}

void KvCacheManager::update_pressure() {
    const PressureLevel now = pressure();
    if (now != last_pressure_) {
        last_pressure_ = now;
        emit({CacheEvent::Kind::pressure_changed, 0, kInvalidBlock, kInvalidBlock, now});
    }
}

std::optional<std::size_t> KvCacheManager::find_checkpoint(const CacheFingerprint& fingerprint,
                                                          ModelArchitecture architecture,
                                                          std::span<const TokenId> tokens,
                                                          std::size_t limit) const {
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < checkpoints_.size(); ++i) {
        const auto& cp = checkpoints_[i];
        if (cp.fingerprint != fingerprint || cp.architecture != architecture || cp.state.position > limit ||
            cp.state.position > tokens.size() || (best && cp.state.position <= checkpoints_[*best].state.position)) {
            continue;
        }
        std::size_t matched = 0;
        for (const auto b : cp.blocks) {
            const auto& block = blocks_[b].tokens;
            const auto take = std::min(cp.state.position - matched, block.size());
            if (!std::equal(block.begin(), block.begin() + static_cast<std::ptrdiff_t>(take),
                            tokens.begin() + static_cast<std::ptrdiff_t>(matched))) break;
            matched += take;
        }
        if (matched == cp.state.position) best = i;
    }
    return best;
}

bool KvCacheManager::checkpoint_uses_block(BlockId block) const noexcept {
    return std::any_of(checkpoints_.begin(), checkpoints_.end(), [block](const Checkpoint& cp) {
        return std::find(cp.blocks.begin(), cp.blocks.end(), block) != cp.blocks.end();
    });
}

void KvCacheManager::erase_checkpoint(std::size_t index) {
    const auto state = checkpoints_[index].state;
    checkpoints_.erase(checkpoints_.begin() + static_cast<std::ptrdiff_t>(index));
    ++counters_.checkpoint_evictions;
    emit({CacheEvent::Kind::checkpoint_evicted, state.owner, kInvalidBlock, kInvalidBlock,
          PressureLevel::normal, state});
}

void KvCacheManager::remove_checkpoints_for_block(BlockId block) {
    for (std::size_t i = checkpoints_.size(); i > 0; --i) {
        const auto& refs = checkpoints_[i - 1].blocks;
        if (std::find(refs.begin(), refs.end(), block) != refs.end()) erase_checkpoint(i - 1);
    }
}

bool KvCacheManager::validate() const {
    const std::size_t bs = config_.block_size_tokens;
    if (checkpoints_.size() > config_.max_checkpoints) return false;
    std::uint64_t bytes = 0;
    for (const auto& cp : checkpoints_) {
        if (cp.state.state_size == 0 || cp.state.state_size > config_.max_checkpoint_bytes - bytes ||
            cp.state.position == 0 || cp.architecture == ModelArchitecture::attention_only ||
            cp.blocks.size() != blocks_for_tokens(cp.state.position)) return false;
        bytes += cp.state.state_size;
        std::size_t remaining = cp.state.position;
        for (const auto b : cp.blocks) {
            if (b >= blocks_.size() || blocks_[b].tokens.size() < std::min(remaining, bs)) return false;
            remaining -= std::min(remaining, bs);
        }
    }
    std::vector<std::uint32_t> refs(blocks_.size(), 0);
    for (const auto& [id, seq] : sequences_) {
        if (seq.blocks.size() != blocks_for_tokens(seq.num_tokens)) return false;
        if (seq.cached_prefix > seq.num_tokens) return false;
        const std::size_t full = seq.num_tokens / bs;
        for (std::size_t i = 0; i < seq.blocks.size(); ++i) {
            const BlockId b = seq.blocks[i];
            if (b >= blocks_.size()) return false;
            ++refs[b];
            const Block& blk = blocks_[b];
            const std::size_t expect = i < full ? bs : seq.num_tokens % bs;
            if (blk.tokens.size() < expect) return false;
            if (i < full && !blk.has_hash) return false;
        }
    }
    std::vector<bool> in_free(blocks_.size(), false);
    for (const BlockId b : free_list_) {
        if (b >= blocks_.size() || in_free[b]) return false;
        in_free[b] = true;
    }
    std::size_t evictable = 0;
    std::size_t registered = 0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        if (blk.ref_count != refs[b]) return false;
        if (blk.registered) ++registered;
        if (in_free[b] && (blk.ref_count != 0 || blk.registered || blk.evictable)) return false;
        if (blk.evictable) {
            if (blk.ref_count != 0 || !blk.registered) return false;
            ++evictable;
        }
        if (blk.ref_count == 0 && !blk.evictable && !in_free[b]) return false;  // leaked
    }
    if (evictable != policy_->size()) return false;
    if (registered != prefix_index_.size()) return false;
    for (const auto& [hash, b] : prefix_index_) {
        const Block& blk = blocks_[b];
        if (!blk.registered || !blk.has_hash || blk.hash != hash || blk.tokens.size() != bs) return false;
    }
    return true;
}

}  // namespace sonder::inference::cache
