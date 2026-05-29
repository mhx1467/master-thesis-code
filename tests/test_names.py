from hsi_compression.utils.names import safe_path_component, safe_sample_stem


def test_safe_path_component_replaces_unsafe_characters():
    assert safe_path_component("model/name with spaces") == "model_name_with_spaces"
    assert safe_path_component("../") == "item"
    assert safe_path_component("", fallback="fallback") == "fallback"


def test_safe_sample_stem_pads_numeric_ids_and_sanitizes_text_ids():
    assert safe_sample_stem("7") == "0007"
    assert safe_sample_stem("sample/with space") == "sample_with_space"
