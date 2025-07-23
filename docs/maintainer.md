# Maintainers Notes

Information relevant for package maintenance

## This section covers the process for making a release from the develop branch

### On the pre-release branch
1. Run the unit tests
2. Merge to main
3. Tag the release in github
4. Prepare the package files

## This section covers asking new contributors to add their details to .zenodo.json

```
Thank you very much for your contribution.

When we release a new version of PyEarthTools, that version is archived on Zenodo. See: https://doi.org/10.5281/zenodo.15760768

As you have contributed to PyEarthtools, would you like to be listed on Zenodo as an author the next time PyEarthTools is archived?

If so, please open a new pull request. In that pull request please add your details to .zenodo.json (which can be found in the PyEarthTools root directory).

In .zenodo.json, please add your details at the bottom of the “creators” section. The fields you will need to complete are:

1. “orcid”. This is an optional field. If you don’t have an ORCID, but would like one, you can obtain one here: https://info.orcid.org/researchers/ .
2. “affiliation”. Options include: the institution you are affiliated with, “Independent Researcher” or “Independent Contributor”.
3. “name”. Format: surname, given name(s).
```

## This section gives guidance for maintaining compatibility with old versions of Python and packages

tldr; about 3 years old is OK, longer if painless

[https://scientific-python.org/specs/spec-0000/](https://scientific-python.org/specs/spec-0000/) provides a guide for the scientific Python ecosystem - we should aspire to be at least that compatible with older versions. It describes an approach including outlining when particular packages move out of support.

We have not tested compatibility against all possible package versions which are included in this spec. Conversely, in some cases, it has been fairly straightforward to support packages older than this.

There is no formal "support" agreement for PyEarthTools. In the context of PyEarthTools package management, maintaining compability means being willing to make reasonable efforts to resolve any issues raised on the issue tracker. If a specific issue arises that would make it impractical to support a version within the compatibility window, then a response will be discussed and agreed on at the time on the basis of practicality.

There is currently no specific testing for older versions of libraries, only older versions of Python (which may or may not intake an older library version). A full matrix test of Python and package versioning would be prohibitively complex, and there would also be no guarantee that pinned older versions wouldn't result in an insecure build (even if only in a test runner).

The development branch versioning is unpinned, and so any issues arising from newly-released packages should quickly be encountered and then resolved before the next PyEarthTools release. Releases of PyEarthTools use "~=" versioning, which gives flexibility within a range of versions (see [https://packaging.python.org/en/latest/specifications/version-specifiers/#id5](https://packaging.python.org/en/latest/specifications/version-specifiers/#id5)).

## This section covers how to build the documentation locally
(Readthedocs should update automatically from a GitHub Action)

### 1. Summary of the tech stack

PyEarthTools utilises:

 - Sphinx, with the myst parts, for making the HTML pages
 - Markdown files as the text source for all the documentation
 - Myst Parser to enable Sphinx to utilise the markdown syntax
 - The 'sphinx book' theme for Sphinx
 - Pandoc should be installed through the OS package manager separately

### 2. Useful information resources are:

 - We follow this recipe: [https://www.youtube.com/watch?v=qRSb299awB0](https://www.youtube.com/watch?v=qRSb299awB0). There isn't a good tutorial written up but the video is excellent.
 - [https://www.sphinx-doc.org/en/master/usage/markdown.html](https://www.sphinx-doc.org/en/master/usage/markdown.html)
 - [https://www.markdownguide.org/cheat-sheet/](https://www.markdownguide.org/cheat-sheet/)

### 3. Process to generate the HTML pages

 - `sphinx-build -b html docs/ htmldocs` is the key command to rebuild the HTML documentation from current sources
 - This requires docs/conf.py to be configured appropriately.
 - Don't forget to separately install pandac through conda

### 4. Generating the markdown for the API documentation

 - Each function must be added explicitly to api.md. The autogeneration tools are not sophisticated enough to process
   the import structure used to define the public API neatly.

## This section covers checking the documentation renders properly in readthedocs

### Common rendering issues in readthedocs

Frequent issues include:

- Lists (including lists that use bullets, dot points, hyphens, numbers, letters etc.)
  - Check **each** list appears and renders properly
  - Check **all** indented lists/sub-lists for proper indentation
- Figures: check **each** figure appears and renders properly
- Plots: check **each** plot appears and renders properly
- Tables: check **each** table appears and renders properly
- Formulae: check **each** formula appears and renders properly
- API Documentation: in addition to checking the above items, also confirm "Returns" and "Return Type" are rendering as expected

### Tutorial rendering

Things that render well in JupyterLab do not always render properly in readthedocs. Additionally, fixes that work well when built locally, don't always work when merged into the codebase.

To check the rendering of tutorials in readthedocs:
  - Compare the tutorial in readthedocs against a version running in JupyterLab (as not everything renders in GitHub).
  - Check the entirety of the tutorial (sometimes things will render properly in one section, while not rendering properly in a different section of the same tutorial).
  - If you make any changes to the code cells, re-execute the Notebook in JupyterLab before committing, otherwise some things (e.g. some plots) won't render in readthedocs. Then re-check the tutorial in readthedocs to ensure the tutorial is still rendering properly.
