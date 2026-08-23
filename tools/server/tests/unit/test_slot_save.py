import pytest
from utils import *
import base64
import requests

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_save_restore_ram():
    global server
    server.slot_ram_limit = -1  # no limit
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
        "store": "ram",
    })
    assert res.status_code == 200
    assert res.body["store"] == "ram"
    assert res.body["n_saved"] == 84
    n_written = res.body["n_written"]
    assert n_written > 0
    assert res.body["ram_store"]["count"] == 1
    assert res.body["ram_store"]["bytes"] == n_written

    # nothing was written to disk
    assert not os.path.exists(os.path.join(server.slot_save_path, "slot1.bin"))

    # move the slot off the saved state
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # the state is still in the store after a restore, so it can be restored again
    for _ in range(2):
        res = server.make_request("POST", "/slots/0?action=restore", data={
            "filename": "slot1.bin",
            "store": "ram",
        })
        assert res.status_code == 200
        assert res.body["store"] == "ram"
        assert res.body["n_restored"] == 84
        assert res.body["ram_store"]["count"] == 1

        res = server.make_request("POST", "/completion", data={
            "prompt": "What is the capital of Germany?",
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        assert res.body["timings"]["prompt_n"] == 6  # only the different part is processed


def test_slot_ram_drop():
    global server
    server.slot_ram_limit = -1
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "scratch.bin",
        "store": "ram",
    })
    assert res.status_code == 200
    n_written = res.body["n_written"]

    res = server.make_request("POST", "/slots/1?action=drop", data={
        "filename": "scratch.bin",
    })
    assert res.status_code == 200
    assert res.body["n_dropped"] == 1
    assert res.body["n_freed"] == n_written
    assert res.body["ram_store"]["count"] == 0
    assert res.body["ram_store"]["bytes"] == 0

    # dropping a name that is gone is not an error
    res = server.make_request("POST", "/slots/1?action=drop", data={
        "filename": "scratch.bin",
    })
    assert res.status_code == 200
    assert res.body["n_dropped"] == 0

    # ... but restoring it is
    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "scratch.bin",
        "store": "ram",
    })
    assert res.status_code != 200
    assert "scratch.bin" in res.body["error"]["message"]


def test_slot_ram_limit_refuses_save():
    global server
    limit_mib = 1
    server.slot_ram_limit = limit_mib
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "fill_0.bin",
        "store": "ram",
    })
    assert res.status_code == 200
    n_written = res.body["n_written"]
    assert res.body["ram_store"]["limit"] == limit_mib * 1024 * 1024

    # saving the same slot again under a new name costs the same, so this many more saves
    # must run into the limit. It has to be refused, never silently evicted
    n_saved = 1
    for i in range(1, limit_mib * 1024 * 1024 // n_written + 2):
        res = server.make_request("POST", "/slots/1?action=save", data={
            "filename": f"fill_{i}.bin",
            "store": "ram",
        })
        if res.status_code != 200:
            break
        assert res.body["ram_store"]["count"] == n_saved + 1
        n_saved += 1
    else:
        assert False, "the RAM slot store never reached its limit"

    assert res.status_code == 503
    assert "limit" in res.body["error"]["message"]

    # overwriting an existing name is still allowed - it replaces, it does not add
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "fill_0.bin",
        "store": "ram",
    })
    assert res.status_code == 200
    assert res.body["ram_store"]["count"] == n_saved

    # falling back to disk still works
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "fallback.bin",
    })
    assert res.status_code == 200
    assert res.body["store"] == "disk"


def test_slot_ram_disabled_by_default():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
        "store": "ram",
    })
    assert res.status_code == 501
    assert res.body["error"]["type"] == "not_supported_error"


def test_slot_save_rejects_unknown_store():
    global server
    server.start()

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
        "store": "nvme",
    })
    assert res.status_code == 400


def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# Regression coverage for issue #21133: slot save/restore/erase must be gated on
# the slot's CONTENT (does it actually hold image/audio tokens) rather than the
# model's CAPABILITY (is an mmproj loaded). A pure-text slot on a multimodal
# server must save/restore/erase normally; a slot that actually holds an image
# must be rejected with ERROR_TYPE_NOT_SUPPORTED (HTTP 501).
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF
    # registry API and auto-downloaded on first run.
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # The restored slot is usable for a follow-up completion. We do NOT assert
    # prefix reuse here: tinygemma3 is a SWA model, which forces full prompt
    # re-processing after a restore (a model property, not the save/restore gate
    # under test).
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_rejected_when_slot_holds_image(mmproj_server):
    server = mmproj_server
    server.start()

    # Process a prompt that actually contains an image on slot 1.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [ _get_img_base64(IMG_URL_CAT) ],
        },
    })
    assert res.status_code == 200

    # Saving a slot that holds image tokens must be rejected (HTTP 501,
    # not_supported_error).
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code != 200
    assert res.body["error"]["type"] == "not_supported_error"


def test_slot_erase_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Erasing a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # Re-running the same prompt should process all tokens again.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == prompt_n  # all tokens are processed again
