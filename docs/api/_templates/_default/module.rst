.. _{{module.rst_label}}:

{{module.canonical_path}}
=====================

.. automodule:: {{module.rst_import_name}}

{% if module.modules %}
.. toctree::
    :maxdepth: 1
    :caption: Modules
    {% for m in module.modules %}
    {{m.api.relative_handle(module.doc_dir)}}{% endfor %}
{% endif %}

{% if module.objects %}
.. toctree::
    :maxdepth: 1
    :caption: Objects
    {% for entry in module.objects %}
    {{module.canonical_path}}.{{entry.name}} <{{entry.api.relative_handle(module.doc_dir)}}>{% endfor %}
{% endif %}

{% if module.classes %}
.. toctree::
    :maxdepth: 1
    :caption: Classes
    {% for entry in module.classes %}
    {{module.canonical_path}}.{{entry.name}} <{{entry.api.relative_handle(module.doc_dir)}}>{% endfor %}
{% endif %}

{% if module.functions %}
.. toctree::
   :maxdepth: 1
   :caption: Functions
   {% for entry in module.functions %}
   {{module.canonical_path}}.{{entry.name}} <{{entry.api.relative_handle(module.doc_dir)}}>{% endfor %}
{% endif %}