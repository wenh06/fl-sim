#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pre_build.py generates pdf files for algorithms in ../fl_sim/algorithms
in standalone mode, which can be included in the documentation,
since sphinx (sphinxcontrib-pseudocode) does not support latex packages like algorithm2e.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from fl_sim.utils.misc import execute_cmd


DOC_BODY = r"""
\documentclass{standalone}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage[ruled,linesnumbered,algosection,nofillcomment]{algorithm2e}

\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\expectation}{\mathbb{E}}
\DeclareMathOperator*{\minimize}{minimize}
\newcommand{\prox}{\mathbf{prox}}
\newcommand{\dom}{\operatorname{dom}}
\newcommand{\col}{\operatorname{col}}

\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}

\renewcommand{\thealgocf}{}

\begin{document}

%s

\end{document}

"""


def main():
    pre_build_dir = (
        Path(__file__).resolve().parent / "source/generated/algorithms/pre-build"
    )
    pre_build_dir.mkdir(parents=True, exist_ok=True)
    # find algorithm folders in ../fl_sim/algorithms
    algorithm_tex_blocks = [
        folder / f"{folder.name}.tex"
        for folder in (
            Path(__file__).resolve().parents[1] / "fl_sim/algorithms"
        ).iterdir()
        if folder.is_dir()
        and f"{folder.name}.tex" in [file.name for file in folder.iterdir()]
    ]

    temp_tex_file = pre_build_dir / "temp.tex"
    with tqdm(algorithm_tex_blocks) as pbar:
        for tex_file in pbar:
            pbar.set_description(tex_file.stem)
            # format tex file using DOC_BODY
            content = DOC_BODY % tex_file.read_text()
            temp_tex_file.write_text(content)
            # use latexmk with xelatex to compile tex file
            # and name the generated pdf file as the same name as the tex file
            cmd = (
                f"""latexmk -xelatex -f -outdir="{str(pre_build_dir)}" """
                f"""-jobname="{tex_file.stem}" "{str(temp_tex_file)}" """
            )
            exitcode, _ = execute_cmd(cmd)
            generated_pdf_file = pre_build_dir / f"{tex_file.stem}.pdf"
            assert generated_pdf_file.exists()
            # move the generated pdf file to ../fl_sim/docs/source/generated/algorithms
            generated_pdf_file.rename(
                generated_pdf_file.parents[1] / f"{tex_file.stem}.pdf"
            )
            generated_pdf_file = generated_pdf_file.parents[1] / f"{tex_file.stem}.pdf"
            # convert the pdf file to svg file using pdf2svg
            generated_svg_file = generated_pdf_file.with_suffix(".svg")
            cmd = f"""pdf2svg "{str(generated_pdf_file)}" "{str(generated_svg_file)}" """
            exitcode, _ = execute_cmd(cmd)
            assert generated_svg_file.exists()
    # remove the temp directory pre_build_dir
    shutil.rmtree(pre_build_dir)


if __name__ == "__main__":
    main()
