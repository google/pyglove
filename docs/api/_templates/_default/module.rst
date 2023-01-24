.. _{{module.rst_label}}:

{{module.canonical_path}}
=====================

.. automodule:: {{module.rst_import_name}}

{% if module.modules %}
Modules
*******
.. toctree::
    :maxdepth: 1
    {% for entry in module.modules %}
    {{entry.name}}<{{entry.api.relative_handle(module.doc_dir)}}>{% endfor %}
{% endif %}

{% if module.objects %}
Objects
*******

.. toctree::
    :maxdepth: 1
    {% for entry in module.objects %}
    {{entry.name}} <{{entry.api.relative_handle(module.doc_dir)}}>{% endfor %}
{% endif %}

{% if module.classes %}
Classes
*******
.. toctree::
    :maxdepth: 1
    {% for entry in module.classes %}
    {{entry.name}} <{{entry.api.relative_handle(module.doc_dir)}}>{% endfor %}
{% endif %}

{% if module.functions %}
Functions
*********
.. toctree::
   :maxdepth: 1
   {% for entry in module.functions %}
   {{entry.name}} <{{entry.api.relative_handle(module.doc_dir)}}>{% endfor %}
{% endif %}