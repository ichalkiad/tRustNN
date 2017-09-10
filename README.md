Given the success of neural networks in recent years, and especially after the success
of deep architectures, their use has been expanding to ever more critical application
areas such as security, autonomous driving, and healthcare. Contrary to previous well-
documented and thoroughly tested approaches, we still have little understanding of what
such models learn and when they could fail. The question that naturally arises is whether
we can trust such systems to undertake safety-critical tasks. Furthermore, in the light
of recent European Union directives (2016 General Data Protection Regulation, art.
22) that essentially require accountable models, companies employing such technologies
should be able to explain them in an understandable way to non-expert customers.

Here we focus on a sentiment classification task and aim to provide a
framework for a data-driven interpretation of the operation of a Long-Short-Term-
Memory Recurrent Neural Network. We believe that, given the difficulty in defining
and measuring the interpretability of neural network models, the evaluation of the latter
should initially focus around users, and later on a rigorous evaluation metric. Therefore,
we provide a critical evaluation of the framework based on our experience and a pilot
study, and set the guidelines for a complete user-based evaluation at a future stage.