from openjarvis.agents.selection_support import (
    presented_items,
    requested_item_kind,
    research_context_marker,
    resolve_follow_up,
    response_needs_repair,
)
from openjarvis.core.types import Message, Role


def test_resolves_only_the_marked_option_set_after_source_listing():
    answer = (
        "1. **Estados Unidos** — variedade de roteiros.\n"
        "2. **Itália** — cultura e gastronomia.\n"
        "3. **Portugal** — cidades históricas e litoral."
    )
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=answer
            + research_context_marker(
                "Pesquise 3 melhores países para turismo.", answer
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="Estas são as fontes usadas na resposta anterior:\n- link 1\n- link 2",
        ),
    ]

    follow_up = resolve_follow_up(
        "Agora compare os três países e explique os pontos positivos e os cuidados.",
        messages,
    )

    assert follow_up.options == ["Estados Unidos", "Itália", "Portugal"]
    assert follow_up.comparison_instruction
    assert "Opções permitidas" in follow_up.prior_research_context


def test_extracts_numbered_bold_items_without_category_headings():
    answer = (
        "### Opções econômicas\n"
        "**1. México** texto.\n**2. Bolívia** texto.\n"
        "### Opções culturais\n**3. Japão** texto."
    )

    assert presented_items(answer) == ["México", "Bolívia", "Japão"]


def test_rejects_a_choice_that_names_an_outside_winner_first():
    assert response_needs_repair(
        "Minha indicação é Brasil, por ser acessível. China ficou atrás.",
        options=["China", "Tailândia", "Portugal"],
        is_choice=True,
        is_comparison=False,
    )


def test_rejects_a_comparison_that_omits_option_labels():
    assert response_needs_repair(
        "Pontos positivos: cultura e paisagens. Cuidados: planeje o orçamento.",
        options=["China", "Tailândia", "Portugal"],
        is_choice=False,
        is_comparison=True,
    )


def test_list_kind_uses_the_numbered_subject_not_incidental_option_word():
    assert (
        requested_item_kind(
            "Pesquise cinco notebooks concretos para programação e explique cada opção."
        )
        == "notebooks"
    )


def test_legacy_generic_marker_is_a_safe_fallback_for_a_typed_reference():
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. **Lenovo IdeaPad** — portátil.\n"
                "2. **Dell Inspiron** — expansível."
                "<!-- openjarvis-research-context:"
                '{"kind":"opcoes","items":["Lenovo IdeaPad","Dell Inspiron"]} -->'
            ),
        )
    ]

    assert resolve_follow_up(
        "Entre esses dois notebooks, qual você indicaria?", messages
    ).options == ["Lenovo IdeaPad", "Dell Inspiron"]


def test_rejects_comparison_missing_a_required_section_part():
    assert response_needs_repair(
        "### Tailândia\nPontos positivos: cultura.\nCuidados: planejamento.\n"
        "### Portugal\nPontos positivos: gastronomia.\n"
        "### Espanha\nPontos positivos: museus.\nCuidados: alta temporada.",
        options=["Tailândia", "Portugal", "Espanha"],
        is_choice=False,
        is_comparison=True,
    )
