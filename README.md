# &nu;DoBe

&nu;DoBe is a Python tool that automates calculations of neutrinoless Double-Beta Decay observables based on an effective field theory (EFT) approach introduced in [arXiv:1806.02780, arXiv:1708.09390].

Users can calculate the expected half-lives, electron spectra, angular correlations between the outgoing electrons and more for any given EFT model based either on the Standard Model EFT (SMEFT) or an Low-Energy EFT (LEFT). To do so, the only inputs required from the user are the Wilson coefficients of the considered EFT model and &nu;DoBe will take care of all subsequent calculations. &nu;DoBe covers all isotopes of experimental interest and includes nuclear matrix elements (NMEs) from the shell model (SM), the interacting boson model (IBM2) and the quasiparticle random phase approximation (QRPA). Additional NMEs can be provided manually by the user if desired.

Please take a look at our <a href="https://arxiv.org/abs/2304.05415">documentation</a> [arXiv:2304.05415 [hep-ph]].

We encourage sharing BSM studies that utilize &nu;DoBe. If you want to contribute a BSM analysis feel free to send us a 
Jupyter notebook to (scholer@mpi-hd.mpg.de) and we will add it to the provided ExampleNotebooks.

<h2> Features </h2>

- SMEFT and LEFT operators up to dimension 9
- Decay Half-Lives
- Single Electron Spectra
- Angular Correlation Coefficients
- Half-Life Ratios
- Operator Limits given experimental limits on the half-life (SMEFT, LEFT, up to dim 9)
- Limits on the parameter space of two simultaneous operators
- Various Plots
- Decay rate formula for given operator combinations
- Operator Matching SMEFT &rarr; LEFT
- RGE running (SMEFT-dim7, LEFT)

<hr>
<h2> &nu;DoBe - Online</h2>
We provide a light version of &nu;DoBe online (<a href="https://nudobe.streamlit.app/">Link</a>). This is intended for quick and dirty studies only as the online version tends to be quite slow. We use Streamlit to generate and host the User Interace (UI). Sometimes, this leads to unexpected crashes of the UI which can be resolved by simply rerunning. Still, we suggest using the full downloadable Python tool for publication purposes.

<hr>
<h2> Requirements</h2>
&nu;DoBe requires Python 3.6 or higher as well as the following 3rd party python packages:

- <a href="https://numpy.org">NumPy</a>
- <a href="https://scipy.org">SciPy</a>
- <a href="https://pandas.pydata.org">Pandas<a>
 - <a href="https://matplotlib.org">Matplotlib</a>
 - <a href="http://mpmath.org/">mpmath</a> (only if PSF scheme B is required)
 
<hr>
<h2> Installation</h2>
To use &nu;DoBe in your Python projects simply download the folder and put it into your projects main directory

<h4> Project tree<h4>
MyProjectDirectory<br>
|<br>
|____MyProject.py<br>
|<br>
|____MyProjectNotebook.ipynb<br>
|<br>
|____nudobe<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        |<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        |____ExampleNotebooks<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        |____NMEs<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        |____PSFs<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        |____src<br>

<hr>
<h2> References</h2>

If you use this tool for your work please make reference to

&nu;DoBe: arXiv:2304.05415 [hep-ph]
 
Additionally, you may consider the underlying works related to

The EFT formalism:
 - Dim-7: arXiv:1708.09390
 - Dim-9: arXiv:1806.02780

NMEs:
 - Shell Model: arXiv:1804.02105 [nucl-th]
 - QRPA: Phys. Rev. C 91 no. 2, (2015) 024613
 - IBM2: arXiv:2009.10119 [hep-ph]
 - CDFT: arXiv:2403.17722 [nucl-th]
 
3rd Party Python Packages:
 - NumPy: Nature 585 no. 7825, (Sept., 2020) 357–362
 - SciPy: Nature Methods 17 (2020) 261–272
 - Pandas: https://doi.org/10.5281/zenodo.4067057, https://doi.org/10.25080/Majora-92bf1922-00a
 - Matplotlib: Computing in Science & Engineering 9 no. 3, (2007) 90–95.
 - mpmath: http://mpmath.org/
 
 # License
&nu;DoBe is published under the <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons NonCommercial-ShareAlike 4.0 International</a> (cc-by-nc-sa-4.0) license. In short, you are allowed to use, redistribute and change the source code as long as you make reference to our work. Additionally, &nu;DoBe is limited to non-commercial usage only. Though, if you find a commercial use-case we would be highly impressed.
