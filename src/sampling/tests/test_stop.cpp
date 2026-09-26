#include <string>
#include <vector>

#include "sonder/sampling/stop.hpp"
#include "sampling_test_util.hpp"

using sonder::inference::sampling::StopSequenceMatcher;

TEST_CASE("no stop sequences streams everything") {
    StopSequenceMatcher m(std::vector<std::string>{});
    const auto r = m.feed("hello");
    CHECK(!r.stopped);
    CHECK(r.emit == "hello");
    CHECK(m.flush().empty());
}

TEST_CASE("match inside a single piece emits the prefix only") {
    StopSequenceMatcher m({"</s>"});
    const auto r = m.feed("abc</s>def");
    CHECK(r.stopped);
    CHECK_EQ(r.stop_index, std::size_t{0});
    CHECK(r.emit == "abc");
    CHECK(m.stopped());
}

TEST_CASE("match spanning pieces holds back the partial prefix") {
    StopSequenceMatcher m({"User:"});
    auto r = m.feed("Hi Us");
    CHECK(!r.stopped);
    CHECK(r.emit == "Hi ");
    r = m.feed("er");
    CHECK(!r.stopped);
    CHECK(r.emit.empty());
    r = m.feed(": more");
    CHECK(r.stopped);
    CHECK(r.emit.empty());
}

TEST_CASE("held text is released when the partial match diverges") {
    StopSequenceMatcher m({"User:"});
    auto r = m.feed("Use");
    CHECK(r.emit.empty());
    r = m.feed("ful");
    CHECK(!r.stopped);
    CHECK(r.emit == "Useful");
}

TEST_CASE("flush returns held-back text at end of generation") {
    StopSequenceMatcher m({"###"});
    const auto r = m.feed("done #");
    CHECK(r.emit == "done ");
    CHECK(m.flush() == "#");
    CHECK(m.flush().empty());
}

TEST_CASE("earliest match wins ties prefer the longest sequence") {
    StopSequenceMatcher m({"cd", "b"});
    auto r = m.feed("abcd");
    CHECK(r.stopped);
    CHECK_EQ(r.stop_index, std::size_t{1});
    CHECK(r.emit == "a");

    StopSequenceMatcher t({"ab", "abc"});
    r = t.feed("xabcz");
    CHECK(r.stopped);
    CHECK_EQ(r.stop_index, std::size_t{1});
    CHECK(r.emit == "x");
}

TEST_CASE("stop at the very start emits nothing") {
    StopSequenceMatcher m({"\n\n"});
    const auto r = m.feed("\n\nrest");
    CHECK(r.stopped);
    CHECK(r.emit.empty());
}

TEST_CASE("feeding after stop keeps reporting stopped - reset re-arms") {
    StopSequenceMatcher m({"x"});
    CHECK(m.feed("x").stopped);
    const auto again = m.feed("more");
    CHECK(again.stopped);
    CHECK(again.emit.empty());
    m.reset();
    CHECK(!m.stopped());
    CHECK(m.feed("abc").emit == "abc");
}

TEST_CASE("empty stop strings are ignored") {
    StopSequenceMatcher m({"", "END"});
    CHECK_EQ(m.stops().size(), std::size_t{1});
    const auto r = m.feed("text");
    CHECK(!r.stopped);
    CHECK(r.emit == "text");
}

TEST_CASE("multi-byte UTF-8 stop sequences split across pieces") {
    const std::string stop = "\xE2\x80\x94" "END";  // em dash + END
    StopSequenceMatcher m({stop});
    auto r = m.feed("a\xE2\x80");
    CHECK(r.emit == "a");
    r = m.feed("\x94" "EN");
    CHECK(!r.stopped);
    r = m.feed("D!");
    CHECK(r.stopped);
    CHECK(r.emit.empty());
}

TEST_CASE("overlapping prefixes hold the longest candidate") {
    StopSequenceMatcher m({"aab"});
    auto r = m.feed("aa");
    CHECK(r.emit.empty());
    r = m.feed("a");  // "aaa": suffix "aa" could still start "aab"
    CHECK(r.emit == "a");
    r = m.feed("b");
    CHECK(r.stopped);
    CHECK(r.emit.empty());
}
