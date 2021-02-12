# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from docutils.nodes import Element
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

class AdmonitionNode(Element):
  """A custom node for info and warning boxes."""


def visit_admonition_html(self, node):
  self.body.append(self.starttag(node, 'div'))


def depart_admonition_html(self, node):
  self.body.append(node['colab_badge_html'])
  self.body.append('</div>\n')


def visit_admonition_latex(self, node):
  self.body.append(f"\n\\begin{{sphinxadmonition}}{node['classes'][1]}{{}}\\unskip")


def depart_admonition_latex(self, node):
  self.body.append('\\end{sphinxadmonition}\n')


class OpenInColab(SphinxDirective):
  """Add an Open In Colab link to a notebook."""

  required_arguments = 0
  optional_arguments = 0
  option_spec = {}
  has_content = True

  def run(self):
    """This is called by the reST parser."""
    html = (
      '<p>Interactive online version: '
      f'<a href="{self.env.config.open_in_colab_base_url}{self.env.docname}.ipynb">'
      f'<img alt="Open In Colab" src="{self.env.config.open_in_colab_badge_url}" '
      'style="vertical-align:text-bottom"></a></p>'
    )
    node = AdmonitionNode(classes=['admonition'], colab_badge_html=html)
    self.state.nested_parse(self.content, self.content_offset, node)
    return [node]

def setup(app):
  app.add_config_value('open_in_colab_base_url', "https://colab.research.google.com/github/google/jax/blob/master/docs/", "html")
  app.add_config_value('open_in_colab_badge_url', "https://colab.research.google.com/assets/colab-badge.svg", "html")
  app.add_directive("open-in-colab", OpenInColab)
  app.add_node(AdmonitionNode,
                html=(visit_admonition_html, depart_admonition_html),
                latex=(visit_admonition_latex, depart_admonition_latex))