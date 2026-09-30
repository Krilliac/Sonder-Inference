#include "priority_admission.hpp"

#include <algorithm>

namespace sonder::inference::server::detail {

void PriorityAdmission::configure(std::size_t capacity, std::size_t subagent, std::size_t background,
                                  std::size_t queue_per_class) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    caps_ = {0, subagent, background};
    queue_cap_ = queue_per_class;
}

bool PriorityAdmission::eligible(std::size_t cls) const {
    return (capacity_ == 0 || total_running_ < capacity_) &&
           (caps_[cls] == 0 || running_[cls] < caps_[cls]);
}

PriorityAdmission::Ticket::Ticket(PriorityAdmission& owner, RequestPriority priority,
                                  std::function<bool()> cancelled, Clock::time_point deadline)
    : owner_(owner), cls_(static_cast<std::size_t>(priority)), cancelled_(std::move(cancelled)), deadline_(deadline) {}

std::shared_ptr<PriorityAdmission::Ticket> PriorityAdmission::enqueue(
    RequestPriority priority, std::function<bool()> cancelled, Clock::time_point deadline) {
    auto ticket = std::shared_ptr<Ticket>(new Ticket(*this, priority, std::move(cancelled), deadline));
    std::lock_guard<std::mutex> lock(mutex_);
    auto& queue = waiting_[ticket->cls_];
    ticket->order_ = next_order_++;
    if (ticket->cancelled_()) return ticket;
    if (Clock::now() >= deadline) {
        ticket->outcome_ = Outcome::expired;
    } else if (queue_cap_ != 0 && queue.size() >= queue_cap_) {
        ticket->outcome_ = Outcome::full;
    } else {
        queue.push_back(ticket.get());
        ticket->queued_ = true;
        // accepted into the admission system; not yet holding a slot
        ticket->outcome_ = Outcome::admitted;
    }
    return ticket;
}

PriorityAdmission::Ticket::~Ticket() { release(); }

bool PriorityAdmission::Ticket::ready_locked() const {
    if (held_) return true;
    if (!queued_ || cancelled_() || Clock::now() >= deadline_) return false;
    if (!owner_.eligible(cls_) || owner_.waiting_[cls_].front() != this) return false;
    for (std::size_t c = 0; c < cls_; ++c) {
        if (!owner_.waiting_[c].empty() && owner_.eligible(c)) return false;
    }
    return true;
}

bool PriorityAdmission::Ticket::ready() const {
    std::lock_guard<std::mutex> lock(owner_.mutex_);
    return ready_locked();
}

bool PriorityAdmission::Ticket::try_locked() {
    if (held_) return true;
    if (!ready_locked()) return false;
    owner_.waiting_[cls_].pop_front();
    queued_ = false;
    held_ = true;
    elapsed_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
    ++owner_.running_[cls_];
    ++owner_.total_running_;
    owner_.cv_.notify_all();
    return true;
}

bool PriorityAdmission::Ticket::try_acquire() {
    std::lock_guard<std::mutex> lock(owner_.mutex_);
    return try_locked();
}

Status PriorityAdmission::Ticket::wait(const CancellationToken& cancel) {
    std::unique_lock<std::mutex> lock(owner_.mutex_);
    for (;;) {
        if (cancel.cancelled() || cancelled_()) return Status(ErrorCode::cancelled, "cancelled while queued");
        if (Clock::now() >= deadline_) return Status(ErrorCode::timeout, "deadline exceeded while queued");
        if (try_locked()) return Status::success();
        if (!queued_) return Status(ErrorCode::unavailable, "admission ticket is no longer queued");
        owner_.cv_.wait_until(lock, std::min(deadline_, Clock::now() + std::chrono::milliseconds(20)));
    }
}

double PriorityAdmission::Ticket::queue_ms() const {
    std::lock_guard<std::mutex> lock(owner_.mutex_);
    return queued_ ? std::chrono::duration<double, std::milli>(Clock::now() - started_).count() : elapsed_ms_;
}

PriorityAdmission::Outcome PriorityAdmission::Ticket::outcome() const {
    std::lock_guard<std::mutex> lock(owner_.mutex_);
    return outcome_;
}

void PriorityAdmission::Ticket::release() {
    std::lock_guard<std::mutex> lock(owner_.mutex_);
    if (held_) {
        --owner_.running_[cls_];
        --owner_.total_running_;
        held_ = false;
    }
    if (queued_) {
        elapsed_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - started_).count();
        auto& queue = owner_.waiting_[cls_];
        const auto it = std::find(queue.begin(), queue.end(), this);
        if (it != queue.end()) queue.erase(it);
        queued_ = false;
    }
    owner_.cv_.notify_all();
}

PriorityAdmission::Result PriorityAdmission::acquire(RequestPriority priority,
                                                      const std::function<bool()>& cancelled,
                                                      Clock::time_point deadline) {
    auto ticket = enqueue(priority, cancelled, deadline);
    if (ticket->outcome() != Outcome::admitted) return {ticket->outcome(), 0.0};
    const Status status = ticket->wait({});
    const double elapsed = ticket->queue_ms();
    if (!status.ok()) {
        return {status.code() == ErrorCode::timeout ? Outcome::expired : Outcome::cancelled, elapsed};
    }
    // The blocking convenience API transfers the permit to the caller, which
    // must release it explicitly. Ticket users keep automatic RAII cleanup.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ticket->held_ = false;
    }
    return {Outcome::admitted, elapsed};
}

void PriorityAdmission::release(RequestPriority priority) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        --running_[static_cast<std::size_t>(priority)];
        --total_running_;
    }
    cv_.notify_all();
}

std::array<std::size_t, 3> PriorityAdmission::queued() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {waiting_[0].size(), waiting_[1].size(), waiting_[2].size()};
}

}  // namespace sonder::inference::server::detail
