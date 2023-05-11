Industrial Uses
===============

Google Cloud Vertex AI
**********************

PyGlove is the `driving force <https://cloud.google.com/vertex-ai/docs/training/neural-architecture-search/pyglove>`_ 
behind `Vertex AI NAS <https://cloud.google.com/vertex-ai/docs/training/neural-architecture-search/overview>`_, which
has brought `advanced AutoML technologies <https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-nas-makes-the-most--advanced-ml-modeling-possible>`_
to industries with significant impact.  Companies such as 
`Qualcomm <https://www.qualcomm.com/news/releases/2021/11/qualcomm-technologies-and-google-cloud-announce-collaboration-neural>`_, 
`Varian <https://www.varian.com/about-varian/newsroom/press-releases/varian-and-google-cloud-collaborate-aid-fight-against-cancer>`_, and 
`Oppo <https://cloud.google.com/blog/products/ai-machine-learning/oppo-leads-with-ai-capabilities-on-mobile-devices>`_
have all benefited from this groundbreaking technology.

Vertex AI has published a few search spaces for computer vision as examples:

  * EfficientNet [`paper <https://arxiv.org/abs/2104.00298>`_][`code <https://github.com/google/vertex-ai-nas/blob/main/nas_architecture/tunable_efficientnetv2_search_space.py>`_]
  * SpineNet [`paper <https://arxiv.org/abs/1912.05027>`_][`code <https://github.com/google/vertex-ai-nas/blob/main/nas_architecture/tunable_spinenet_search_space.py>`_]
  * MNASNet [`paper <https://arxiv.org/abs/1807.11626>`_][`code <https://github.com/google/vertex-ai-nas/blob/main/nas_architecture/tunable_mnasnet_search_space.py>`_]
  * NASFpn [`paper <https://arxiv.org/abs/1904.07392>`_][`code <https://github.com/google/vertex-ai-nas/blob/main/nas_architecture/tunable_nasfpn_search_space.py>`_]
  * AutoAugment [`paper <https://arxiv.org/abs/1906.11172>`_][`code <https://github.com/google/vertex-ai-nas/blob/main/nas_architecture/tunable_autoaugment_search_space.py>`_]

Pax 
***

Pax is a powerful machine learning framework developed by Google, based on Jax, that is designed for training large-scale models.
It utilizes PyGlove to enable its hyperparameter tuning and AutoML capabilities. Pax serves as a good example of how PyGlove can
be seamlessly integrated into a large scale ML codebase based on dynamic evaluation:

* `Inspecting the search space <https://github.com/google/paxml/blob/ad16d2b52e6460ed66d2f00d64ace6338b0f2b57/paxml/tuning_lib.py#L56>`_
* `Implementing the tuning loop <https://github.com/google/paxml/blob/ad16d2b52e6460ed66d2f00d64ace6338b0f2b57/paxml/tuning_lib.py#L222>`_

Vizier
******

`Vizier <https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf>`_
is the distributed tuning solution at Alphabet. PyGlove uses it as a backend for serving distributed AutoML scenarios at Google.
The `open-source Vizier <https://github.com/google/vizier>`_ has shown how PyGlove can be
`used together with Vizier <https://oss-vizier.readthedocs.io/en/latest/advanced_topics/pyglove/vizier_as_backend.html>`_,
it also serve as an example on how PyGlove backend could be developed. 

* `Implementing the Backend interface <https://github.com/google/vizier/blob/60c2430ee10fb8e75075a9a9eba15c1258c6ad58/vizier/_src/pyglove/backend.py#L67>`_
* `Implementing the Feedback interface <https://github.com/google/vizier/blob/60c2430ee10fb8e75075a9a9eba15c1258c6ad58/vizier/_src/pyglove/core.py#L114>`_

