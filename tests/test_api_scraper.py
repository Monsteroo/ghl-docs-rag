from scraper.api_scraper import parse_api_spec

SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "paths": {
        "/contacts/search": {
            "post": {
                "operationId": "search-contacts-advanced",
                "summary": "Search Contacts",
                "description": "Search contacts based on combinations of advanced filters.",
                "parameters": [
                    {"name": "Version", "in": "header", "required": True, "schema": {"type": "string"}}
                ],
                "tags": ["Search"],
            }
        },
        "/contacts/search/duplicate": {
            "get": {
                "operationId": "get-duplicate-contact",
                "summary": "Get Duplicate Contact",
                "description": "Get Duplicate Contact by email or phone.",
                "parameters": [
                    {"name": "locationId", "in": "query", "required": True, "schema": {"type": "string"}},
                    {"name": "email", "in": "query", "required": False, "schema": {"type": "string"}},
                ],
                "tags": ["Search"],
            }
        },
    },
}


def test_parses_one_chunk_per_path_and_method():
    chunks = parse_api_spec(SAMPLE_SPEC, module="contacts")
    assert len(chunks) == 2
    ids = {c["doc_id"] for c in chunks}
    assert ids == {"api-contacts-search-contacts-advanced", "api-contacts-get-duplicate-contact"}


def test_chunk_title_is_the_summary():
    chunks = parse_api_spec(SAMPLE_SPEC, module="contacts")
    by_id = {c["doc_id"]: c for c in chunks}
    assert by_id["api-contacts-search-contacts-advanced"]["title"] == "Search Contacts"


def test_chunk_content_includes_method_path_and_params_not_just_description():
    chunks = parse_api_spec(SAMPLE_SPEC, module="contacts")
    by_id = {c["doc_id"]: c for c in chunks}
    content = by_id["api-contacts-get-duplicate-contact"]["content"]
    assert "GET" in content
    assert "/contacts/search/duplicate" in content
    assert "locationId" in content
    assert "Get Duplicate Contact by email or phone." in content


def test_two_methods_on_the_same_path_dont_collide():
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/contacts/{id}": {
                "get": {"operationId": "get-contact", "summary": "Get Contact", "description": "d", "parameters": [], "tags": []},
                "delete": {"operationId": "delete-contact", "summary": "Delete Contact", "description": "d", "parameters": [], "tags": []},
            }
        },
    }
    chunks = parse_api_spec(spec, module="contacts")
    assert len(chunks) == 2
    assert {c["doc_id"] for c in chunks} == {"api-contacts-get-contact", "api-contacts-delete-contact"}
