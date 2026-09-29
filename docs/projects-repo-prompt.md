# Prompt: turn `~/Projects` into the `Projects` repo

Paste this into Claude Code started from `~/Projects` with `claude --dangerously-skip-permissions`.
It builds one private GitHub repo, `MrBlueStudios/Projects`, that mirrors the folder tree. Every
project that already is a git (or jj) repo stays its own repo and is linked in as a submodule.
Then it adds this repo at `Projects/Laya_And_CLM-MCPs`.

Before running it, rename this repo on GitHub (Settings → General → Repository name →
`Laya_And_CLM-MCPs`). Step 0 of the prompt does the rename with `gh` if you haven't.

````text
Goal: put everything in ~/Projects (3d-projects, Blitsol, Music, Research, Roblox-Games,
ZEN_Projects) on GitHub as ONE private repo, MrBlueStudios/Projects, whose tree mirrors this
folder. Many folders are already git repos, some jj (Jujutsu) repos, and some of those contain
further nested repos, several levels deep. Each existing repo must stay its own independent repo
with its own history and remote; the Projects repo links to them as git submodules at their
current paths. Plain folders that aren't repos are tracked directly in Projects.

You're running with --dangerously-skip-permissions, so these rules are hard limits, not
suggestions:
- Never delete, move or rename a .git or .jj directory, and never run rm -rf, git clean,
  git reset --hard, git checkout -- . or git stash drop anywhere under ~/Projects.
- Never rewrite history or force-push, in any repo.
- Never commit other people's uncommitted work in a nested repo: list it and ask me.
- Never push to a remote I don't own (anything not under github.com/MrBlueStudios). Third-party
  clones are linked as submodules at their upstream URL, as they are.
- Never commit secrets: .env files, private keys, tokens, credential JSON, cookies, databases
  with personal data. If one is already committed in a nested repo, tell me; don't fix history.
- New GitHub repos are always private.

Before you start: read this whole prompt, then check it against what you actually find in
~/Projects. If a step is wrong, risky or missing for this layout (for example a better choice
than submodules for a specific folder), tell me and propose the fix before doing it.

Step 0. Preconditions.
- `gh auth status` must show MrBlueStudios with the repo scope; if not, stop and tell me to run
  `gh auth login`. Check git, gh, jj (if any .jj dirs exist) and git-lfs are installed.
- The old MrBlueStudios/Projects repo holds the Laya/CLM MCP servers, not my projects. Check
  with `gh repo view MrBlueStudios/Projects --json name,url` and
  `gh api repos/MrBlueStudios/Projects/contents --jq '.[].name'`: if it still exists and lists
  laya-mcp-server, rename it with
  `gh repo rename Laya_And_CLM-MCPs --repo MrBlueStudios/Projects --yes`.
  Make sure MrBlueStudios/Laya_And_CLM-MCPs exists and MrBlueStudios/Projects no longer does.
  If ~/Projects is itself already a git or jj repo, stop and show me its status and remotes.

Step 1. Inventory (read-only). Walk ~/Projects to any depth, without descending into
node_modules, .venv, venv, __pycache__, build, dist, target, .gradle, Library/Temp caches.
For every git or jj repo you find, record:
  path, git or jj (colocated or not), current branch or bookmark, remotes and whether I own them,
  whether HEAD is pushed (commits ahead of upstream, or no upstream at all), uncommitted changes,
  untracked files, submodules it already declares, repos nested inside it that it does NOT
  declare, and its size.
For folders that are not inside any repo, record size, the largest files, and file types.
Also find, everywhere: files over 50 MB (GitHub hard limit is 100 MB), likely secrets (use
gitleaks if installed, else grep for key/token patterns and file names like .env, *.pem, id_rsa,
*credentials*.json), symlinks that point outside ~/Projects, and empty directories.
Write all of this to ~/projects-inventory.md (outside ~/Projects, so it never gets committed).

Step 2. Plan, then stop and wait for my "go". Write the plan into the same inventory file and
summarise it for me:
- Which folders become submodules (with URL) and which are tracked directly.
- Repos with no remote: propose a private MrBlueStudios repo for each (name = folder name, or
  parent-folder prefixed if two would clash) so they can be submodules. Ask if any should instead
  be absorbed as plain files.
- Repos whose current commit isn't on their remote: they have to be pushed first (submodules
  point at a commit; if it isn't pushed, a fresh clone breaks). List them and the branch you'd
  push. Uncommitted work: list it, don't touch it.
- Nested repos a parent repo doesn't declare: work bottom-up, and propose adding them as
  submodules inside that parent (a new commit in the parent) or adding them to the parent's
  .gitignore. Show the exact change per parent.
- Large binaries (Music audio, 3D files like .blend/.fbx/.obj, Roblox .rbxl/.rbxm, video, PSDs):
  propose per folder whether to track them with Git LFS, ignore them, or keep them local only.
  Mention my GitHub LFS storage/bandwidth quota and estimate the total LFS size.
- The .gitignore for the Projects repo: OS/editor junk, dependency and build dirs, *.blend1,
  logs, and the secrets you found.
- Anything else surprising.
Do nothing that writes until I reply "go" (or change the plan).

Step 3. Execute, after "go", in this order, checking each step before the next:
a. Bottom-up, for each nested repo: create its GitHub repo if the plan says so
   (`gh repo create MrBlueStudios/<name> --private --source <path> --remote origin`; for a
   non-colocated jj repo create it without --source and run `jj git remote add origin <url>`), push the
   branch the plan names (git push -u origin <branch>; for jj: jj git push --bookmark <name>),
   and apply the planned .gitmodules / .gitignore change to its parent repo with a clear commit
   message, then push that parent.
b. In ~/Projects: git init -b main, write .gitignore and a README.md with a table of the
   projects (folder, one-line description taken from each project's own README, repo link).
c. Add each top-level repo in place with `git submodule add <url> <path>` (git accepts an
   existing clone at that path). Add the plain folders with git add, using git lfs track first
   where the plan said LFS.
d. Before committing, check the staged set: `git status`, `git diff --cached --stat`, no file
   over 50 MB outside LFS, and re-run the secret scan on staged files only. Fix and re-check.
e. Commit with a clear message ("Create Projects repo: ..." plus a short body listing the
   submodules and tracked folders), create the private repo with
   `gh repo create MrBlueStudios/Projects --private --source . --remote origin --push`.

Step 4. Add Laya_And_CLM-MCPs.
  `git submodule add https://github.com/MrBlueStudios/Laya_And_CLM-MCPs.git Laya_And_CLM-MCPs`,
  add it to the README table, commit ("Add Laya_And_CLM-MCPs as a submodule") and push.
  If uv is installed, run ./Laya_And_CLM-MCPs/register-decision-mcps.sh so the MCP servers are
  registered from their new path, and check with `claude mcp list`.

Step 5. Verify and report.
- Clone MrBlueStudios/Projects with --recurse-submodules into a temp dir outside ~/Projects and
  check every submodule checks out (no missing-commit errors), then delete only that temp dir.
- `git submodule status --recursive` in ~/Projects shows no "-" or "+" entries you didn't expect.
- Report: repos created, branches pushed, parent repos you committed to, folders on LFS or
  ignored, anything you skipped or that needs me (uncommitted work, secrets, big files), and how
  to clone everything on a new machine (git clone --recurse-submodules ...), and how to update
  the submodule pointers later (commit in the project, then `git add <path>` in Projects).
````
