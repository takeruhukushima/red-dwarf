instructions = """
1. Create a release issue.
2. Create a release prep feature branch.
    i. Update CHANGELOG.md
        - Add the new version heading below "Unreleased".
        - Write "_No changes yet._" in "Unreleased" section.
        - Add the link for the new version tag. (bottom of changelog)
        - Update the "Unreleased" comparison link (bottom) to the new tag.
    ii. Update `pyproject.yaml` to new version.
    iii. Commit the changes as "Prepare for release vX.Y.Z"
    iv. Merge release prep branch
3. Create a tagged commit and push:
    $ git tag vX.Y.Z -m "Release vX.Y.Z"
4. Build the package artifacts:
    $ uv build
5. Create a GitHub release.
    - See: https://github.com/polis-community/red-dwarf/releases/new
    - Choose the tag `vX.Y.Z`
    - Set the title `vX.Y.Z`
    - Copy the new changelog text into the body.
    - Attach the wheel and tar builf files from `dist/`
6. Publish to PyPI:
    $ uv publish
7. Announce in Polis User Group discord.
    - See: https://discord.com/invite/wFWB8kzQpP
"""

print(instructions.strip())
