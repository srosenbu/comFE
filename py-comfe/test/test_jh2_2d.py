import basix
import comfe as co
import dolfinx as dfx
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from scipy.spatial import KDTree


def y_i(pressure, parameters):
    p_s = pressure / parameters["PHEL"]
    t_s = parameters["T"] / parameters["PHEL"]
    return parameters["SIGMAHEL"] * parameters["A"] * np.power(p_s + t_s, parameters["N"])


def y_f(pressure, parameters):
    p_s = pressure / parameters["PHEL"]
    t_s = parameters["T"] / parameters["PHEL"]
    return parameters["SIGMAHEL"] * parameters["B"] * np.power(p_s, parameters["M"])


case1_parameters = {
    "RHO": 3.7e-6,
    "SHEAR_MODULUS": 90.16,
    "A": 0.93,
    "B": 0.0,
    "C": 0.0,
    "M": 0.0,
    "N": 0.6,
    "EPS0": 1.0,
    "T": 0.2,
    "SIGMAHEL": 2.0,
    "PHEL": 1.46,
    "D1": 0.0,
    "D2": 0.0,
    "K1": 130.95,
    "K2": 0.0,
    "K3": 0.0,
    "BETA": 1.0,
    "EFMIN": 0.0,
}
case1_points = np.array(
    [
        -0.007919648983332861,
        0.005099160183002827,
        0.14482222565429526,
        0.19329716153501453,
        0.21739537733532144,
        0.310299341463971,
        0.3300934002081184,
        0.46798846485908996,
        0.4990391602336395,
        0.6917995680651807,
        0.6438613858879347,
        0.8850916658860113,
        0.7804803945646093,
        1.0478546308932142,
        0.9254646590727833,
        1.2615028597325857,
        1.0381221722321108,
        1.4141029503730698,
        1.1585778052343616,
        1.546341846280829,
        1.2956424207592026,
        1.7650841715890195,
        1.4802654398975106,
        1.9583509508389314,
        # The following entries are not in the original data
        # 1.5420022432253835,
        # 1.7140368689029724,
        # 1.596629591839318,
        # 1.576597538528536,
        # 1.6590955700096468,
        # 1.4238860461760101,
        # 1.689315816258067,
        # 1.2203044811338672,
        # 1.8207090718971455,
        # 0.7265822207931296,
        # 1.8514559444206729,
        # 0.5891580815612452,
        2.006000501307704,
        0.003818040494522723,
        4.529451827621042,
        0.012390908607554874,
        7.458668698903958,
        -0.004739636475958875,
    ]
).reshape(-1, 2)

case2_parameters = {
    "RHO": 3.7e-6,
    "SHEAR_MODULUS": 90.16,
    "A": 0.93,
    "B": 0.0,
    "C": 0.0,
    "M": 0.0,
    "N": 0.6,
    "EPS0": 1.0,
    "T": 0.2,
    "SIGMAHEL": 2.0,
    "PHEL": 1.46,  # 38695149172,
    "D1": 0.005,
    "D2": 1.0,
    "K1": 130.95,
    "K2": 0.0,
    "K3": 0.0,
    "BETA": 1.0,
    "EFMIN": 0.0,
}

case2_points = np.array(
    [
        0,
        0.000006169145634782325,
        0.2548103913089075,
        0.33505246858362536,
        0.5184056460020847,
        0.6814561651356899,
        0.7206055633355333,
        0.959721648148947,
        1.0195870373911915,
        1.3799638488065782,
        1.1865487948573996,
        1.6014361771038326,
        1.3535599054886887,
        1.8285902885309406,
        1.537548504907555,
        2.0102777966279444,
        1.720500687859738,
        2.072647858998007,
        1.9813321653084262,
        2.100871700278228,
        2.2767602114783116,
        2.1120255155863457,
        2.545981726990628,
        2.106152488941806,
        2.6499194926494614,
        2.0719877604150594,
        3.04005626260819,
        1.9864834019136683,
        3.265156048538837,
        1.9010962571793426,
        3.654601874186443,
        1.7360469348599905,
        3.85339642312936,
        1.6222693819132983,
        3.982997834629881,
        1.5426318809108122,
        4.337008087749926,
        1.2980622713560392,
        4.6736953799268335,
        1.059186783222391,
        4.941238887826424,
        0.860133130162803,
        5.182526511903366,
        0.638370852020703,
        5.423715429650147,
        0.40524500761889426,
        5.604644132834043,
        0.23466196166492814,
        5.707989660511914,
        0.13231583557993076,
        5.8370975403616345,
        -0.004139496721098812,
        6.123740723147251,
        -0.004343078527054622,
        7.57432895118355,
        -0.005373325848103239,
    ]
).reshape(-1, 2)
case3_parameters = {
    "RHO": 3.7e-6,
    "SHEAR_MODULUS": 90.16,
    "A": 0.93,
    "B": 0.31,
    "C": 0.0,
    "M": 0.6,
    "N": 0.6,
    "EPS0": 1.0,
    "T": 0.2,
    "SIGMAHEL": 2.0,
    "PHEL": 1.46,
    "D1": 0.005,
    "D2": 1.0,
    "K1": 130.95,
    "K2": 0.0,
    "K3": 0.0,
    "BETA": 1.0,
    "EFMIN": 0.0,
}
case3_points = np.array(
    [
        0,
        0,
        0.17209426228334257,
        0.24258760107816713,
        0.46449410073013997,
        0.6361185983827498,
        0.6882853877420074,
        0.9595687331536391,
        0.9719544786615146,
        1.3261455525606471,
        1.1525961656894226,
        1.574123989218329,
        1.4190787457284362,
        1.9191374663072778,
        1.4879622813374542,
        2.0215633423180597,
        1.6499050445273853,
        2.070080862533693,
        1.8118936384129976,
        2.123989218328841,
        2.033393390640799,
        2.1778975741239894,
        2.2377066319881074,
        2.2102425876010785,
        2.450383975297255,
        2.226415094339623,
        2.731074070997477,
        2.2425876010781676,
        2.9266565648174936,
        2.2479784366576823,
        3.2580354099412503,
        2.226415094339623,
        3.597778357026847,
        2.1886792452830193,
        3.8523678715365604,
        2.134770889487871,
        4.090000028644185,
        2.0862533692722374,
        4.437923754908897,
        2.01078167115903,
        4.675280927842434,
        1.9299191374663076,
        4.997562379873449,
        1.8382749326145555,
        5.285745794317567,
        1.7358490566037739,
        5.539922832566146,
        1.6334231805929922,
        5.785644107461524,
        1.5363881401617252,
        6.031457043748263,
        1.4501347708894878,
        6.227177029655325,
        1.4716981132075473,
        6.542194316420851,
        1.5256064690026958,
        6.780513933963697,
        1.5579514824797847,
        7.129400104837716,
        1.5956873315363882,
        7.5549381142387375,
        1.6495956873315365,
        7.245626749085535,
        1.2668463611859841,
        6.936177891845288,
        0.8679245283018866,
        6.678151075159475,
        0.5175202156334233,
        6.274176408076514,
        0,
        5.82031502874444,
        0.6145552560646901,
        5.467327010606941,
        1.0943396226415096,
        5.324862293081569,
        1.3369272237196768,
        4.7802103056048075,
        1.2722371967654988,
        4.345987379372176,
        1.1967654986522913,
        3.7415034186834557,
        1.0943396226415096,
        3.0517743640274864,
        0.9649595687331538,
        2.3109440836868504,
        0.8247978436657681,
        1.595343601318778,
        0.6522911051212938,
        0.9818539089286782,
        0.4905660377358494,
        0.6663783152063383,
        0.3827493261455528,
    ]
).reshape(-1, 2)

case3a_parameters = {
    "RHO": 3.7e-6,
    "SHEAR_MODULUS": 90.16,
    "A": 0.93,
    "B": 0.31,
    "C": 0.0,
    "M": 0.6,
    "N": 0.6,
    "EPS0": 1.0,
    "T": 0.2,
    "SIGMAHEL": 2.0,
    "PHEL": 1.46,
    "D1": 0.00815,
    "D2": 1.0,
    "K1": 130.95,
    "K2": 0.0,
    "K3": 0.0,
    "BETA": 1.0,
    "EFMIN": 0.0,
}
case3a_points = np.array(
    [
        0,
        -0.005976054661711139,
        0.32526550724505277,
        0.44796423504901206,
        0.8733446488496777,
        1.1885906699531925,
        1.3192460485166089,
        1.7918432336486603,
        1.4956401757179376,
        2.01880309859348,
        1.6886831891480691,
        2.0604915624778695,
        2.1023467893554955,
        2.1498239850872762,
        2.368564490447678,
        2.1675534041352655,
        2.5796270448455383,
        2.1733718335020002,
        2.8089284449693537,
        2.167224446997924,
        3.1114593589447788,
        2.143094070319157,
        3.4321194662670447,
        2.095045768446157,
        3.7984315140937577,
        2.023058981557841,
        4.164633909541355,
        1.939120085346102,
        4.5855528421668215,
        1.8192837415219294,
        4.942306857614101,
        1.7054714252753305,
        5.308125469734801,
        1.5797001464316147,
        5.5916499713304715,
        1.4838708203597057,
        5.8019449590745324,
        1.4060244844624863,
        6.2429120016813355,
        1.4714321286039653,
        6.67459514285649,
        1.524894516695717,
        7.299266471272218,
        1.6140693140101474,
        7.528896828533376,
        1.643778255476337,
        7.222418428909848,
        1.2376326965119122,
        6.925333583097085,
        0.8553845029206366,
        6.646652061578949,
        0.4790986574436831,
        6.25700233239748,
        0.007278176663689351,
        5.867334327819493,
        0.5334656776631248,
        5.6315451619314345,
        0.8324465958647345,
        5.35955243553925,
        1.185239419116522,
        5.259841872131618,
        1.3167880076848044,
        5.085311835375228,
        1.2930140012381581,
        4.6809869627890075,
        1.22160288934019,
        4.258368418287275,
        1.1561815386513214,
        3.725494406586453,
        1.072914263261656,
        3.3211147078106755,
        0.9955270967019763,
        2.916735009034899,
        0.918139930142297,
        2.466319586427777,
        0.8228588659659581,
        1.7493026794015718,
        0.6680160001096525,
        1.280429106310266,
        0.5607965331572804,
        0.884785047070568,
        0.4355940760302186,
        0.6824581322087873,
        0.3819603560961018,
    ]
).reshape(-1, 2)

case_1 = {"parameters": case1_parameters, "points": case1_points}
case_2 = {"parameters": case2_parameters, "points": case2_points}
case_3 = {"parameters": case3_parameters, "points": case3_points}
case_3a = {"parameters": case3a_parameters, "points": case3a_points}


@pytest.mark.parametrize(
    "test_case",
    [
        case_1,
        case_3a,
    ],
)
def test_single_element_2d(test_case: dict, plot: str | None = None) -> None:
    mesh = dfx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0, 0], [1000.0, 1000.0]]),
        [1, 1],
        cell_type=dfx.mesh.CellType.quadrilateral,
    )
    rule = co.helpers.QuadratureRule(
        quadrature_type=basix.QuadratureType.Default,
        cell_type=basix.CellType.quadrilateral,
        degree=1,
    )

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    P1 = dfx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    u = dfx.fem.Function(P1)
    t_end = 100.0
    v_bc = -50.0 / t_end
    domains = [
        lambda x: np.isclose(x[0], 0.0),
        lambda x: np.isclose(x[0], 1000.0),
        lambda x: np.isclose(x[1], 0.0),
        lambda x: np.isclose(x[1], 1000.0),
    ]
    values = [0.0, 0.0, 0.0, v_bc]
    subspaces = [0, 0, 1, 1]
    boundary_facets = [dfx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, domain) for domain in domains]
    bc_dofs = [
        dfx.fem.locate_dofs_topological(P1.sub(i), mesh.topology.dim - 1, facet)
        for facet, i in zip(boundary_facets, subspaces)
    ]

    bcs = [dfx.fem.dirichletbc(np.array(value), dofs, P1.sub(i)) for value, dofs, i in zip(values, bc_dofs, subspaces)]

    parameters = test_case["parameters"]
    law = co.laws.PyJH23D(parameters)

    v_ = ufl.TestFunction(P1)
    u_ = ufl.TrialFunction(P1)

    h = 1e-1

    mass_form = ufl.inner(u_, v_) * parameters["RHO"] * ufl.dx

    M = dfx.fem.petsc.assemble_matrix(dfx.fem.form(mass_form))
    M.assemble()
    ones = dfx.fem.Function(P1)
    with ones.vector.localForm() as ones_local:
        ones_local.set(1.0)
    M_action = M * ones.vector
    M_function = dfx.fem.Function(P1)
    M_function.vector.array[:] = M_action.array

    M_action.array[:] = 1.0 / M_action.array
    M_action.ghostUpdate()

    solver = co.cdm.CDMPlaneStrain(
        P1,
        0,
        None,
        bcs,
        M_function,
        law,
        rule,
        additional_output=["mises_stress", "pressure", "equivalent_plastic_strain"],
    )
    solver.model.input["density"].vector.array[:] += parameters["RHO"]
    solver.model.output["density"].vector.array[:] += parameters["RHO"]
    s_eq_ = []
    p_ = []

    while solver.t < t_end:
        solver.step(h)
        u_ = max(abs(solver.fields["u"].vector.array))
        p_.append(solver.q_fields["pressure"].vector.array[0])
        s_eq_.append(solver.q_fields["mises_stress"].vector.array[0])

    values = [0.0, 0.0, 0.0, -v_bc]
    subspaces = [0, 0, 1, 1]

    bcs = [dfx.fem.dirichletbc(np.array(value), dofs, P1.sub(i)) for value, dofs, i in zip(values, bc_dofs, subspaces)]
    solver.bcs = bcs

    while solver.t < 2.0 * t_end:  # and counter <= 2000:
        solver.step(h)
        u_ = max(abs(solver.fields["u"].vector.array))
        p_.append(solver.q_fields["pressure"].vector.array[0])
        s_eq_.append(solver.q_fields["mises_stress"].vector.array[0])

    p_ = np.array(p_).reshape((-1, 1))
    s_eq_ = np.array(s_eq_).reshape((-1, 1))

    if plot is not None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg", force=True)
        p_debug = np.linspace(0.0, 8.0, 100)
        # plt.plot(p_debug, y_i(p_debug, parameters))
        # plt.plot(p_debug, y_f(p_debug, parameters))
        plt.plot(p_, s_eq_)
        plt.scatter(test_case["points"][:, 0], test_case["points"][:, 1])
        plt.xlabel("Pressure [GPa]")
        plt.ylabel("Equiv. Stress [GPa]")
        plt.title(f"JH2 test")
        plt.savefig(f"{plot}.png")
        plt.clf()

    points = np.hstack((p_, s_eq_))
    tree = KDTree(points)
    distances = tree.query(test_case["points"])
    assert np.mean(distances[0] / np.max(np.abs(test_case["points"][:, 1]))) < 0.05


if __name__ == "__main__":
    test_single_element_2d(case_3a, plot="jh2_case_3a")
    test_single_element_2d(case_1, plot="jh2_case_1")
    test_single_element_2d(case_2, plot="jh2_case_2")
