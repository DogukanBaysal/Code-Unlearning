from evalplus.eval import untrusted_check


def test_untrusted_check_accepts_timeout_from_environment(monkeypatch):
    monkeypatch.setenv("EVALPLUS_TIMEOUT_PER_TASK", "30")
    monkeypatch.setenv("EVALPLUS_MAX_MEMORY_BYTES", "-1")

    status, details = untrusted_check(
        dataset="humaneval",
        code="def add(a, b):\n    return a + b\n",
        inputs=[(1, 2)],
        entry_point="add",
        expected=[3],
        atol=0,
        ref_time=[0.001],
        fast_check=True,
    )

    assert status == "pass"
    assert details == [True]
