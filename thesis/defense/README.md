# Defense presentation

Beamer scaffold for the TFM defense.

Compile from this directory:

```bash
latexmk -pdf main.tex
```

The generated PDF is written to `build/main.pdf`.

Main files:

- `main.tex`: deck preamble, metadata, section inputs, bibliography.
- `sections/`: editable slide placeholders.
- `figures/`: selected final figures for the defense deck.

The deck reuses the thesis bibliography from `../references.bib` and the UC3M logo from `../figures/logo_UC3M.png`.
