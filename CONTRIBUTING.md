# Contributing

## Your project folder and branch

```
git checkout -b <your-name>
cp -r projects/_template projects/<project-name>
```

Commit your work on your branch as often as you like:

```
git add projects/<project-name>
git commit -m "Sweep detector efficiency"
git push -u origin <your-name>
```

## Adding a function to the package

If you write something everyone could use (a new plot, a new kind of source,
a new protocol), move it into `src/bellgame/`.

1. Put the function in the module where it fits (`plots.py` for plots,
   `link.py` for photonics, and so on).
2. Follow the house style:
   * a short function with a docstring and a small runnable `Example:`
   * plain dicts, lists, and numpy arrays in and out
   * units in parameter names (`distance_km`, `angle_deg`, `coherence_time_ms`)
   * a `seed=` argument if anything is random
   * clear error messages (`raise ValueError("distance_km must be positive, got -3")`)
   * no classes unless there's no other way
3. Export it in `src/bellgame/__init__.py` so `bg.<name>` works.
4. Add a test in `tests/` that checks something you know is true, e.g. a case
   you can work out by hand.
5. Run the checks:

   ```
   uv run pytest
   uv run ruff check src tests tutorial examples projects
   ```

6. Open a pull request from your branch.

## Running everything

```
uv run pytest                                  # tests, including the docstring examples
uv run python tutorial/05_parameter_studies.py # any tutorial part or example
```
