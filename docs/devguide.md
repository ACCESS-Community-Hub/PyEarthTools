# PyEarthTools Core Developer Guide

This is a summary only, and should be expanded to provide more detail. This is intended to provide succinct guidance to the team of people directly involved in coding the PyEarthTools framework. Other guides will be produced for other contributor demographics (such as those involved in producing documentation, tutorials, or other kinds of contribution).

## Quick Overview

- Please fork the repository and do your development on your fork. Create pull requests from your fork.
- Unit test coverage on new pull requests should meet or extend existing levels of test coverage.


## Coding Guidelines

- Use 'black' and 'isort' with a line length of 120 characters

## Nice-to-haves

- Type hints would be nice, but the development so far hasn't required them. Do your best.
- Pylint checking is a good idea, please turn it on and do your best.


## Creating Your Own Fork  of PyEarthTools for the First Time

Unless you are an advanced Git user, we would recommend you follow this process:

1. First (i.e. **before** cloning the PyEarthTools repository) create your own fork using the GitHub web user interface.
2. Clone **your fork**. (Do not directly clone https://github.com/ACCESS-Community-Hub/PyEarthTools).
3. Immediately create a new local branch, with a command such as `git checkout -b branch_name`.


## Workflow for Submitting Pull Requests

Prior to developing a pull request, it may be a good idea to create a GitHub issue to capture what the pull request is trying to achieve, any pertinent details, and (if applicable) how it aligns to the roadmap. Otherwise, please explain this in the pull request.

To submit a pull request, please use the following workflow: 

1. Ensure you are working on a new feature branch in **your fork**.
2. Keep your feature branch rebased and up-to-date with the develop branch of PyEarthTools. You can do this by first syncing the develop branch on your fork, and then rebase your feature branch against the develop branch on your fork.
3. When ready, submit a pull request to the develop branch of https://github.com/ACCESS-Community-Hub/PyEarthTools.

To help disambiguate branches, some contributors like to prefix their branch names with a short numerical indentifier. This is up to the contributor and any approach to branch naming is welcome. 

## Pull Request Etiquette

In general, the originator of a pull request will be the person who does all the coding work, including responding to feedback from others. Typically, feedback will be provided in the form of comments or code suggestions made through the GitHub web user interface.

Sometimes, it may be pragmatic for the package maintainers to make or request a more complex code change. This typically occurs when a complex Git operation is needed to keep a pull request (PR) up to date, to resolve conflicts, or to make changes where the originator of the PR does not know how to proceed. It can also occur in the final stages of a PR lifecycle if there are small tidyups needed and time is a factor.

Not every possibility can be accounted for, and the package maintainers will (if needed) push code directly to a PR to get something over the line. However, special circumstances aside, the maintainers will first leave a comment on the PR asking how the originator would like to proceed, so that there are no surprises.

Most of these kinds of code change can also be handled as a PR by the reviewer onto the fork of the originator. This is slightly slower (i.e. does not take effect immediately), but allows for more control by the originator. This is probably most developer's general preference.

In short - once you have made a PR, the maintainers may then take it, modify it, or include it as-is. However, every effort will be made to communicate about that process and make sure that the originator of the PR is happy with any modifications made.

### Scope of a Code Review

A code review is responsible for checking the following:

1. Unit test coverage is 100% and unit tests cover functionality and robustness (or improves the previous situation to these ends)
2. Any security issues are resolved and appropriately handled
3. Documentation and tutorials are written to cover any new functionality
4. Style guidelines are followed, static analysis and lint checking have been done
5. Code is readable and well-structured
6. Code does not do anything unexpected or beyond the scope of the function
7. Any additional dependencies are justified and do not result in bloat