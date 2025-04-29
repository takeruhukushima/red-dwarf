instructions = """
1. Checkout `main` branch.
2. Edit and stage files (no commit yet).
    i. Update CHANGELOG.md
        - Add the new version heading below "Unreleased".
        - Write "_No changes yet._" in "Unreleased" section.
        - Add the link for the new version tag. (bottom of changelog)
        - Update the "Unreleased" comparison link (bottom) to the new tag.
    ii. Update `pyproject.yaml` to new version.
3. Delete `dist/` directory.
4. Build the package artifacts:
    $ uv build
5. Publish to PyPI:
    $ uv publish
6. Commit change as "Release vX.Y.Z" and push.
7. Create a GitHub release.
    - See: https://github.com/polis-community/red-dwarf/releases/new
    - Name the new tag `vX.Y.Z`
    - Set the title `vX.Y.Z`
    - Copy the new changelog text into the body.
    - Attach the wheel and tar builf files from `dist/`
8. Announce in Polis User Group discord.
    - See: https://discord.com/invite/wFWB8kzQpP
"""

print(instructions.strip())
