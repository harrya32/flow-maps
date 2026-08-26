from common import maizels


EXPECTED_OLD_EDGES = [
    ("NMP", "Mesoderm"),
    ("NMP", "Early_Neural"),
    ("Early_Neural", "Neural"),
    ("Early_Neural", "pMN"),
    ("Early_Neural", "p3"),
    ("p3", "V3"),
    ("p3", "FP"),
    ("pMN", "MN"),
]

EXPECTED_ACTIVE_EDGES = [
    ("NMP", "Mesoderm"),
    ("NMP", "Early_Neural"),
    ("Early_Neural", "Neural"),
    ("Neural", "pMN"),
    ("pMN", "MN"),
    ("Neural", "p3"),
    ("p3", "V3"),
    ("p3", "FP"),
]


def test_old_lineage_is_preserved_and_new_lineage_is_active():
    assert maizels.TRANSITION_EDGES_OLD == EXPECTED_OLD_EDGES
    assert maizels.TRANSITION_EDGES == EXPECTED_ACTIVE_EDGES


def test_active_lineage_direct_transitions():
    reachable = maizels.build_transition_reachable(mode="direct")

    assert reachable["NMP"] == {"NMP", "Mesoderm", "Early_Neural"}
    assert reachable["Early_Neural"] == {"Early_Neural", "Neural"}
    assert reachable["Neural"] == {"Neural", "pMN", "p3"}
    assert reachable["pMN"] == {"pMN", "MN"}
    assert reachable["p3"] == {"p3", "V3", "FP"}

    for terminal_state in ("Mesoderm", "MN", "V3", "FP"):
        assert reachable[terminal_state] == {terminal_state}


def test_active_lineage_descendant_transitions():
    reachable = maizels.build_transition_reachable(mode="descendant")

    assert reachable["Neural"] == {
        "Neural",
        "pMN",
        "MN",
        "p3",
        "V3",
        "FP",
    }
    assert {"pMN", "MN", "p3", "V3", "FP"} <= reachable["Early_Neural"]
    assert "p3" not in reachable["pMN"]
    assert "FP" in reachable["p3"]
