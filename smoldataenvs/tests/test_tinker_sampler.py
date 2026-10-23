import json

from pirl.tinker_sampler import normalize_messages, parse_tool_calls

TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "limit": {"type": "integer"}}}}},
         {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
             "command": {"type": "string"}}}}}]


def test_parse_single_tool_call_with_typed_args():
    text = ("Let me look.\n<tool_call>\n<function=read>\n<parameter=path>\n/home/user/input/a.csv\n</parameter>\n"
            "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>")
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "Let me look."
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/home/user/input/a.csv", "limit": 20}


def test_parse_multiline_command_and_two_calls():
    cmd = "python3 -c \"\nimport pandas as pd\nprint(pd.read_csv('a.csv').shape)\n\""
    text = (f"<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=bash>\n<parameter=command>\nhead -3 a.csv\n</parameter>\n</function>\n</tool_call>")
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "" and len(calls) == 2
    assert json.loads(calls[0]["function"]["arguments"])["command"] == cmd


def test_plain_text_has_no_tool_calls():
    assert parse_tool_calls("The answer is 42.", TOOLS) == ("The answer is 42.", [])


def test_normalize_decodes_arguments_and_content_parts():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}]
    out = normalize_messages(msgs)
    assert out[0]["content"] == "hi" and out[1]["content"] == ""
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
