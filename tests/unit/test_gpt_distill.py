import pytest
from wm.distill.gpt_distill import (
    GPTDistillPipeline,MultiModelDistill,_parse_json)
from wm.distill.claude_distill import DistillQuestion,DistillAnswer,build_distill_rows

def test_parse_json_plain():
    d=_parse_json('{"a":1}')
    assert d=={"a":1}

def test_parse_json_fenced():
    d=_parse_json('```json\n{"a":1}\n```')
    assert d=={"a":1}

def test_parse_json_preamble():
    d=_parse_json('Here is the JSON:\n{"a":1}')
    assert d=={"a":1}

def test_multi_model_distill_no_keys():
    mm=MultiModelDistill()
    rows=mm.run_sync("finance",5)
    assert rows==[]

def test_multi_model_distill_init():
    mm=MultiModelDistill(anthropic_key="fake",openai_key="fake")
    assert mm._ak=="fake"
    assert mm._ok=="fake"

def test_gpt_prompts_exist():
    from wm.distill.gpt_distill import GPT_THINK_ANSWER,GPT_WEBSEARCH_PROMPT,GPT_CODE_PROMPT
    assert "{domain}" in GPT_THINK_ANSWER
    assert "{domain}" in GPT_WEBSEARCH_PROMPT
    assert "{domain}" in GPT_CODE_PROMPT
