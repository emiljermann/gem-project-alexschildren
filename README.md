# gem_project_alexschildren

<details>
<summary>Project Structure</summary>

```
gem_project_alexschildren/
├── e2/             # Main control system
├── dev_utils/      # Small test scripts and experimental code used during development
├── diagrams/       # Diagrams for development planning
├── tutorials/      # Environment and framework setup guides
├── plots/          # Error plots used during development and testing
├── bash_scripts/   # Scripts for module testing and deployment
├── README.md       # Project overview and setup instructions
└── .gitignore      # Git ignore rules
```
</details>
<h2>Notes</h2>

3/28
- Use GNSS to check if vehicle has arrived at destination point
  - Use lane following to continually drive until reached point
- Follow diagram 3 plan 


4/14
- choose median depth unless it is greater than 2 sd different than mean, otherwise use adjusted average.

4/15
- plot error between where aregg is vs where we think he is
- plot point where we guess person to be in gnss
- george bash script to revolutionize your ros development
