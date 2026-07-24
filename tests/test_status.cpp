#include <gtest/gtest.h>

#include "cedar/status.hpp"

#include <utility>

using namespace li;

namespace {

struct Counted {
    static int alive;
    int v;

    explicit Counted(int x) : v(x) { ++alive; }
    Counted(const Counted& o) : v(o.v) { ++alive; }
    Counted(Counted&& o) noexcept : v(o.v) { ++alive; }
    Counted& operator=(const Counted&) = default;
    Counted& operator=(Counted&&) = default;
    ~Counted() { --alive; }
};

int Counted::alive = 0;

}

TEST(Status, ResultCarriesValue) {
    Result<int> r(42);

    EXPECT_TRUE(r.ok());
    EXPECT_EQ(r.status(), Status::ok);
    EXPECT_EQ(r.value(), 42);
}

TEST(Status, ResultCarriesStatus) {
    Result<int> r(Status::not_found);

    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status(), Status::not_found);

    Result<int> c(Status::capacity);

    EXPECT_EQ(c.status(), Status::capacity);
}

TEST(Status, CopyBothStates) {
    Result<int> v(7);
    Result<int> vc(v);

    EXPECT_TRUE(vc.ok());
    EXPECT_EQ(vc.value(), 7);

    Result<int> s(Status::out_of_bounds);
    Result<int> sc(s);

    EXPECT_FALSE(sc.ok());
    EXPECT_EQ(sc.status(), Status::out_of_bounds);
}

TEST(Status, AssignAcrossStates) {
    Result<int> v(7);
    Result<int> s(Status::not_found);

    v = s;

    EXPECT_FALSE(v.ok());
    EXPECT_EQ(v.status(), Status::not_found);

    Result<int> v2(9);
    s = v2;

    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.value(), 9);
}

TEST(Status, MoveValueOut) {
    Result<Counted> r(Counted(5));

    ASSERT_TRUE(r.ok());

    Counted got = std::move(r).value();

    EXPECT_EQ(got.v, 5);
}

TEST(Status, NoLeaksNoDoubleDestroy) {
    {
        Result<Counted> a(Counted(1));
        Result<Counted> b(Status::not_found);
        Result<Counted> c(a);
        Result<Counted> d(std::move(a));

        b = c;
        c = Result<Counted>(Status::capacity);
        d = d;

        Result<Counted> e(Counted(2));
        e = Result<Counted>(Counted(3));

        EXPECT_EQ(e.value().v, 3);
    }

    EXPECT_EQ(Counted::alive, 0);
}

TEST(StatusDeath, LiCheckAlwaysAborts) {
    EXPECT_DEATH(CEDAR_CHECK(false, "boom %d", 7), "CEDAR_CHECK failed");
}

TEST(StatusDeath, LiAssertAbortsUnderInvariantChecks) {
    EXPECT_DEATH(CEDAR_ASSERT(1 == 2), "CEDAR_ASSERT failed");
}
