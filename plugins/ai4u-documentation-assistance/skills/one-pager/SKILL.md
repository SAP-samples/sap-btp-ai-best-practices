---
name: "one-pager"
description: "Create an AI4U use-case one-pager as a single high-resolution PNG image (3840x2160 by default, at least 1920x1080) in the standard AI4U template: title, two personas with photos, names and roles, the app screenshot on a laptop, and Challenge / Solution / Benefits panels. Gathers the content by scanning the current codebase or, in an empty folder, by interviewing the user. Use whenever the user asks for a one-pager, one pager, use-case summary slide, AI4U slide or solution poster for an application, PoC or agent."
---

# One-pager

Produce one PNG that looks like the AI4U one-pagers from the shared deck: same background, logos, laptop, persona circles and panels, with new text, two personas and a screenshot of the app. No PowerPoint is created or edited.

The workflow is **install tools → gather → pick personas → propose → approve → screenshot → render → look → deliver**.

## Rules

- **Nothing is rendered before the user approves the copy.** Show every slot first (step 4).
- **Never invent facts.** Numbers (FTE, $, %, time saved) only when the user or the repository states them. When the code can't answer something, ask.
- **No customer names** on the image: no company names, customer codes, system IDs or real people. Personas get generic first names from the picker. If the repository or the user's notes contain a customer name, describe the business role instead.
- **Fixed template.** Keep the structure, background and subtitle default. Only the slots in [references/copy-guide.md](references/copy-guide.md) change.

## Step 0: install the tools (once per project)

The tools sit next to this file (`<skill-dir>` is the base directory shown when this skill loads). Copy them into the project; don't run or edit them in `<skill-dir>`.

```bash
mkdir -p one-pager/.tools && TOOLS="$(pwd)/one-pager/.tools"
cp -R "<skill-dir>/scripts" "<skill-dir>/assets" "$TOOLS/"
pip install pillow   # add --break-system-packages if pip refuses
```

The renderer looks for the SAP 72 fonts (`72-Regular`, `72-Bold`, `72Brand-Regular`, `72Brand-Bold`) in the system font folders. If they are missing it prints a warning and uses a fallback font; tell the user, and pass `--font-dir` if they have the fonts elsewhere.

| Tool | Purpose |
|---|---|
| `scripts/pick_personas.py [--seed N]` | Two random, distinct persona photos with first names that match each photo |
| `scripts/render_one_pager.py --content one-pager/content.json --output one-pager/one-pager.png [--width 1920] [--screenshot PATH\|clipboard] [--font-dir DIR]` | Render the PNG. Exit code 2 = a text slot overflows; the message names the slot and how many words to cut |
| `scripts/check_one_pager.py` | Self-check: renders the bundled example at 4K and 1080p, overflow and width checks, picker check |
| `scripts/layout.py` | Slot boxes, fonts, colors, laptop screen and circle positions (1920 basis) |
| `assets/example_content.json` | Complete content file to copy from |
| `assets/personas/` | 73 headshot cut-outs, `manifest.json` (gender per photo), `names.json` (name pools) |

## Step 1: gather the content

**Codebase present** (README, `docs/`, `api/`, `ui/`, agent or app code): read the README, the main entry points, prompts and UI routes. Work out the business process, who uses the app, what goes in, what comes out, and who benefits. Note anything you could not determine.

**Empty folder, or the code doesn't explain the business case:** interview the user. Ask in one message, and accept short answers:

1. Use-case name (becomes the title) and the app or agent name.
2. Who does the work today (role) and who owns the outcome (role).
3. What is painful today: sources, manual steps, delays, errors.
4. What the app does: input, AI capability, output, who decides.
5. Three benefits, with figures only if they have measured or agreed ones.

Draft every slot following [references/copy-guide.md](references/copy-guide.md) (voice, word limits, worked examples).

## Step 2: pick the personas

```bash
python one-pager/.tools/scripts/pick_personas.py
```

It returns two different photos and a first name for each that matches the photo (the manifest tags each photo's presented gender; names never repeat). Use them as `left_persona` and `right_persona`; the roles come from the use case. If the user dislikes a pick, run it again (or with another `--seed`), or use a photo they supply (`"photo": "path/to/photo.png"`) and a name that fits it.

## Step 3: propose and wait for approval

Show the user a table with every slot: title, subtitle, app name, left name / role / caption, right name / role / caption, challenge, solution, the three benefits, and the two photo file names. Say which facts came from the code, which from the user, and what you assumed. Wait for approval or edits. Apply the edits, and show the table again if they changed much.

## Step 4: get the screenshot

Always offer the user these options:

1. **They provide it** (simplest): a file path, or they copy the screenshot to the clipboard and reply "done" (render with `--screenshot clipboard`; works on macOS and Windows). Save it to `one-pager/screenshot.png` either way so later re-renders don't depend on the clipboard.
2. **You capture it** from the running UI: start the app (or use the URL the user gives you), open the main screen in the browser tools at about 1920x1080, wait for data to load, take the screenshot and save it to `one-pager/screenshot.png`. Show it to the user before using it, because it can contain real data.
3. **Neither is possible right now:** render without one. The laptop shows "Screenshot pending"; tell the user clearly and re-render once they have it.

The screenshot is scaled to the laptop screen width and cropped at the bottom (or padded) so the app's navigation and actions on both sides stay visible. A wide, header-first screen works best.

## Step 5: render

Write the approved copy to `one-pager/content.json` (same keys as `assets/example_content.json`; relative paths resolve against the JSON file's folder), then:

```bash
python one-pager/.tools/scripts/render_one_pager.py --content one-pager/content.json --output one-pager/one-pager.png
```

Default output is 3840x2160. Use `--width 1920` when the user wants a smaller 1080p file. Anything below 1920 is refused.

If the script exits with an overflow, shorten the named slot by about the number of words it reports, tell the user what you cut, and render again. Don't loosen `layout.py` to squeeze text in.

## Step 6: look, then deliver

Open the PNG (Read tool) and check:

- No text overlaps the laptop, arrows, circles or panel edges; nothing is cut off.
- The first word of each benefit is magenta, and all three benefits start with the same verb form.
- Names suit the photos; roles and captions read left to right as user → app → stakeholder.
- The screenshot is readable at a glance, with no personal or customer data in it.
- No customer names anywhere.

Deliver the path to `one-pager/one-pager.png` and keep `content.json` for later edits. If you changed the tools in `one-pager/.tools` and the change would help every project (a better fit rule, a new slot), tell the user it is a candidate update for this skill.
