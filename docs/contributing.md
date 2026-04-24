# How to Contribe to PyEarthTools

If you are thinking about contributing some code to PyEarthTools, you are very welcome. There are many ways you can help to improve PyEarthTools, for example

- Raise bugs and feature requests in the [GitHub issue tracker](https://github.com/ACCESS-Community-Hub/PyEarthTools/issues)
- Improve documentation and tutorials based on your own experience
- Contribute to code quality and test coverage

[A pinned issue](https://github.com/ACCESS-Community-Hub/PyEarthTools/issues/174) contains a non-exhaustive list of things you could consider doing. Alternatively, you are very welcome to look through the GitHub issue tracker and see if there is anything that grabs your interest. If you are going to work on an issue, please leave a comment on the issue tracker (so we can avoid people doubling up).

## Code of Conduct

Please review our [Code of Conduct](https://github.com/ACCESS-Community-Hub/PyEarthTools/blob/develop/CODE_OF_CONDUCT.md) before contributing and help us create a welcoming and safe environment for everyone.

## Getting Started

To learn more about PyEarthTools in general, a good place to start is the [New Users Guide](https://pyearthtools.readthedocs.io/en/latest/newuser.html) .

If you think you may want to contribute code, you will need to follow the PyEarthTools installation instructions [here](https://pyearthtools.readthedocs.io/en/latest/installation.html#developer-installation) and read the developer guide [here](https://pyearthtools.readthedocs.io/en/latest/devguide.html) .


## Contribution Workflow

Details of the contribution workflow can be found in the [Developer Guide](https://pyearthtools.readthedocs.io/en/latest/devguide.html), including

- Forking the repository in GitHub
- Submitting changes via pull requests from your own fork
- Code reviews
- Coding formatting and test coverage requirements
- GitHub actions

## Generative AI Usage

Generative AI tools can be helpful, but contributors must be transparent about using them in the `PyEarthTools` repository.

1.	Contributors must declare any use of generative AI tools in preparing their pull request. Because `PyEarthTools` may be used for academic work and many journals now require authors to declare if generative AI tools are used (see [1](https://joss.readthedocs.io/en/latest/policies.html#ai-usage-policy ), [2](https://www.ametsoc.org/ams/publications/ethical-guidelines-and-ams-policies/author-disclosure-and-obligations/), [3](https://www.egu.eu/news/1031/statement-on-the-use-of-ai-based-tools-for-the-presentation-and-publication-of-research-results-in-earth-planetary-and-space-science/), [4](https://rmets.onlinelibrary.wiley.com/hub/ai-policy)), it is important to keep track of generative AI usage.
2.	The name of any generative AI tool or system used, and the version, should be included in each pull request, including identifying exactly where they were applied. The pull request template will include this in the checklist.
3.	A human must be in the loop for all pull requests. Human contributors must understand their code, have reviewed it before submitting the pull request, and be able to explain all changes during review.
4.	Contributors are responsible for all submitted content, regardless of whether generative AI tools were used.
5.	If bots wish to propose improvements to the `PyEarthTools` package, they should create an issue rather than submitting a pull request. The issue should clearly describe the proposed feature or change and may include example code snippets to illustrate how the implementation could be improved.
6.	All contributions must adhere to the [code of conduct]([https://github.com/nci/scores/blob/develop/CODE_OF_CONDUCT.md](https://github.com/ACCESS-Community-Hub/PyEarthTools?tab=coc-ov-file)).
7.	Given that generative tools are evolving rapidly, this policy will likely be adjusted over time.

[1] https://joss.readthedocs.io/en/latest/policies.html#ai-usage-policy
[2] https://www.ametsoc.org/ams/publications/ethical-guidelines-and-ams-policies/author-disclosure-and-obligations/
[3] https://www.egu.eu/news/1031/statement-on-the-use-of-ai-based-tools-for-the-presentation-and-publication-of-research-results-in-earth-planetary-and-space-science/
[4] https://rmets.onlinelibrary.wiley.com/hub/ai-policy


## Contributor Recognition in Zenodo

When we release a new version of PyEarthTools, that version is archived on Zenodo. See: [https://doi.org/10.5281/zenodo.15760768](https://doi.org/10.5281/zenodo.15760768).

Once you have contributed to PyEarthtools, you may like to be listed on Zenodo as an author the next time PyEarthTools is archived. If so, add your details to [`.zenodo.json`](https://github.com/ACCESS-Community-Hub/PyEarthTools/blob/develop/.zenodo.json), at the bottom of the “creators” section. The fields you will need to complete are:

1. “orcid”. This is an optional field. If you don’t have an ORCID, but would like one, you can obtain one here: [https://info.orcid.org/researchers/](https://info.orcid.org/researchers/) .
2. “affiliation”. Options include: the institution you are affiliated with, “Independent Researcher” or “Independent Contributor”.
3. “name”. Format: surname, given name(s).

Submit this change, either as a new pull request or as part of another pull request you may be submitting.
