# Dependency migration

The old single-file environment is replaced by dependency groups.

The ML stack pins PyTorch and PyG together because compatibility matters. Other packages use bounded ranges so the project does not immediately become obsolete.

Once a clean Mac environment and Docker environment both pass the tests, we should generate a lock/snapshot from that known-good environment and commit it.

Do not start model development until the environment check and test suite pass.
