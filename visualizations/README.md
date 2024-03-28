# Visualizations & Results


This folder contains the results of vertex normal prediction and Transformer training.

We provide visualizations for vertex normal prediction on mesh objects (`vertex_normal_prediction.pdf`), as well as acceleration direction (`acceleration_prediction.gif`) prediction on deformable objects. 

1. We add two additional baselines, Bartal Tree [1] and FRT Tree [2] to the mesh vertex normal prediction experiment.

2. We add an aditional experiments for acceleration direction prediction on deformable object (flag). As we can see, the predicted acceleration direction from our FTFI can accurately predict the ground truth direction.


For ViT Transformer training using FTFI on ImageNet, we provide the training accuracy curve and compare it with linear attention ViT Performer [3] baseline (`ftfi_vit_imagenet.pdf`). We observe that FTFI achieves <b>7%</b> relative accuracy improvement over the baseline.


[1] Bartal, Y. On approximating arbitrary metrics by tree metrics. In Vitter, J. S. (ed.), Proceedings of the Thirtieth Annual ACM Symposium on the Theory of Computing

[2] Fakcharoenphol, J., Rao, S., and Talwar, K. A tight bound on approximating arbitrary metrics by tree metrics. J. Comput. Syst. Sci.

[3] Rethinking Attention with Performers; K. Choromanski, ICLR 2021.