# Avila et al manuscript repository

This repository contains codes, data, output, and figures from Avila et al manuscript.

*Repository structure:*

- Data: isotopic measurement used in the analyses
- Figure: figures generated in the numerical models
- Output: output files (csv, asvii, npz) from numerical models
- Scripts: python scripts and jupyter notebook to run numerical models

## General Workflow
Steps to make changes in the main script:

1. update your LOCAL/MAIN branch by clicking "Fetch origin" and "Pull origin"
2. make a new branch with an intuitive name (e.g. update-readme, tweak-grenvile, etc)
3. make changes in your LOCAL/NEW-BRANCH 
4. commit changes in your LOCAL/NEW-BRANCH
5. publish your LOCAL/NEW-BRANCH to main repo in github
6. if you are not done with your changes, but there are updates in ORIGIN/MAIN branch, rebase your LOCAL/NEW-BRANCH with ORIGIN/MAIN to keep it up to date and easier for future merge
7. if you are done and happy with your changes make a pull request in github to incorporate your changes to ORIGIN/MAIN
8. delete LOCAL/NEW-BRANCH and ORIGIN/NEW-BRANCH 
8. repeat step 1 to 8

note: ORIGIN = remote repository / GitHub 