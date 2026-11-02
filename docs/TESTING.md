# Testing and verification policy

## Test layers

Use tests at several levels, choosing the narrowest test that verifies the behaviour being changed.

| Test layer | Purpose |
|---|---|
| Unit | Verify a function or class in isolation |
| Contract/schema | Verify input and output structures |
| Integration | Verify communication across software components |
| API | Verify request validation, response shape, and error handling |
| UI smoke | Verify that the interface can load and navigate |
| Reproducibility | Verify that documented setup and build steps work from a clean checkout |

## General rules

- Tests must be deterministic where practical.
- Use small synthetic fixtures for ordinary software tests.
- Keep test fixtures clearly labelled and separate from real source datasets.
- Do not rely on network availability for unit tests.
- Do not place secrets or personal data in fixtures.
- Assert error handling as well as successful paths.
- Avoid tests that pass only because a generated artifact from a previous run is present.

## Running tests

From the repository root:

```bash
python -m pytest
```

For a focused test, pass the relevant test file or test node to pytest.

## Recording results

A test report should include:

- Commit SHA.
- Python and operating-system versions.
- Test command.
- Number of passed, failed, skipped, and errored tests.
- Any external services or datasets required.
- Link to the corresponding CI run, if one exists.

A test file being present is not evidence that it passed. A CI run on an older commit is not evidence that the current commit passed.

## Continuous integration

A general-purpose CI workflow should perform safe, bounded checks on proposed changes. Separate these checks from any scheduled or resource-intensive workflow so that ordinary code review does not depend on an unrelated long-running process.

The current workflow configuration should be reviewed by the repository maintainer before adding or changing automation.
