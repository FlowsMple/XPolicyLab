import numpy as np
import pandas as pd
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_delta_ee_actions_roundtrip():
    rng = np.random.default_rng(0)

    def random_poses(shape):
        pose = rng.normal(size=(*shape, 8)).astype(np.float32)
        quat = pose[..., 3:7]
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        quat[quat[..., 0] < 0] *= -1.0
        return pose

    state = np.concatenate([random_poses(()), random_poses(())], axis=-1)
    actions = np.concatenate([random_poses((5,)), random_poses((5,))], axis=-1)

    delta = _transforms.DeltaEEActions()({"state": state, "actions": actions.copy()})["actions"]
    # Grippers stay absolute, positions become offsets.
    for base in (0, 8):
        np.testing.assert_allclose(delta[..., base + 7], actions[..., base + 7])
        np.testing.assert_allclose(delta[..., base : base + 3], actions[..., base : base + 3] - state[base : base + 3])

    restored = _transforms.AbsoluteEEActions()({"state": state, "actions": delta})["actions"]
    np.testing.assert_allclose(restored, actions, atol=1e-5)


def test_delta_ee_actions_identity_state():
    # An identity state must leave the actions untouched.
    state = np.array([0, 0, 0, 1, 0, 0, 0, 0.3] * 2, dtype=np.float32)
    actions = np.tile(np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 0.7] * 2, dtype=np.float32), (3, 1))

    delta = _transforms.DeltaEEActions()({"state": state, "actions": actions.copy()})["actions"]
    np.testing.assert_allclose(delta, actions, atol=1e-6)


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_extract_prompt_from_task_dataframe():
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=["Task zero", "Hello, world!"])
    transform = _transforms.PromptFromLeRobotTask(tasks)

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"
