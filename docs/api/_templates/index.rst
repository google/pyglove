.. currentmodule:: pyglove

Public API: pyglove
===================

Modules
-------

core
^^^^

.. toctree::
    :maxdepth: 1
{% for e in module.modules -%}
  {%- if e.api.source_category == 'core' %}
    {{e.api.relative_handle(module.doc_dir)}}
  {%- endif -%}
{%- endfor %}

ext
^^^
.. toctree::
    :maxdepth: 1
{% for e in module.modules -%}
  {%- if e.api.source_category == 'ext' %}
    {{e.api.relative_handle(module.doc_dir)}}
  {%- endif -%}
{%- endfor %}

.. toctree::
   :maxdepth: 1

{% for e in module.modules -%}
  {%- if e.api.source_category == 'generators' %}
    {{e.api.relative_handle(module.doc_dir)}}
  {%- endif -%}
{%- endfor %}


Top-level shortcurts
--------------------

Objects
^^^^^^^

{% for e in module.objects %}
* :ref:`pg.{{e.name}}<{{e.api.rst_label}}>`
{%- endfor %}

Classes
^^^^^^^

{% for e in module.classes %}
* :ref:`pg.{{e.name}}<{{e.api.rst_label}}>`
{%- endfor %}

Functions
^^^^^^^^^
{% for e in module.functions %}
* :ref:`pg.{{e.name}}<{{e.api.rst_label}}>`
{%- endfor %}
