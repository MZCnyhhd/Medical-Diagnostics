# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Medical-Diagnostics'
copyright = '2026, ma_zhen_chen'
author = 'ma_zhen_chen'
release = '0.1'

import os
import sys
# 指向 Medical-Diagnostics 这一层，这样 Sphinx 才能看到并导入 'src'
sys.path.insert(0, os.path.abspath('../../'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # 自动从代码提取文档
    'sphinx.ext.napoleon',     # 如果你用了 Google 或 NumPy 风格的注释，需要这个
    'sphinx.ext.viewcode',     # 在文档里查看源代码链接
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
