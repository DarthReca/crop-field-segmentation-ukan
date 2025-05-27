<div align="center">
  
## KAN You See It? KANs and Sentinel for Effective and Explainable Crop Field Segmentation

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Eleonora Poeta**](https://github.com/eleonorapoeta)<sup>1</sup> · [**Eliana Pastor**](https://elianap.github.io/)<sup>1</sup>

[**Tania Cerquitelli**](https://smartdata.polito.it/members/tania-cerquitelli)<sup>1</sup> · [**Elena Baralis**](https://smartdata.polito.it/members/elena-baralis/)<sup>1</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy

**[ECCV 2024 CVPPA Workshop](https://cvppa2024.github.io/)**

<a href="https://arxiv.org/abs/2408.07040"><img src='https://img.shields.io/badge/arXiv-KAN%20You%20See%20It-red' alt='Paper PDF'></a>
<a href='https://beta.source.coop/repositories/radiantearth/south-africa-crops-competition/description/'><img src='https://img.shields.io/badge/Source%20Cooperative-South%20Africa%20Crop%20Type-silver'></a>
<a href="https://huggingface.co/datasets/DarthReca/south-africa-crop-type-clouds"><img src='https://img.shields.io/badge/HuggingFace-Cloud_Masks-yellow?logo=huggingface'></a>
</div>

**This paper analyzes the integration of KAN layers into the U-Net architecture (U-KAN) to segment crop fields using Sentinel-2 and Sentinel-1 satellite images and provides an analysis of the performance and explainability of these networks**. Our findings indicate a 2% improvement in IoU compared to the traditional full-convolutional U-Net model in **fewer GFLOPs**. Furthermore, gradient-based explanation techniques show that U-KAN predictions are highly plausible and that the network has a very high ability to **focus on the boundaries of cultivated areas** rather than on the areas themselves. The per-channel relevance analysis also reveals that some channels are irrelevant to this task.

### Getting Started

Install the dependencies of the *requirements.txt* file. Make sure to edit the config files in the `configs/` folder. Then, simply run *main.py* to train the models.
Use the *xai.ipynb* for the explainability part.

## Contributors
The repository setup is by [Eleonora Poeta](https://github.com/eleonorapoeta) for the XAI section and [Daniele Rege Cambrin](https://github.com/DarthReca) for the remaining.

## Metadata

You can find the computed cloud masks for Sentinel-2 on [HuggingFace](https://huggingface.co/datasets/DarthReca/south-africa-crop-type-clouds).

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

U-Net is licensed under **GPL-3 license**. See [LICENSE](models/UNET_LICENSE) for more information.

U-KAN is licensed under **MIT license**. See [LICENSE](models/UKAN_LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@inbook{RegeCambrin2025,
  title = {KAN You See It? KANs and Sentinel for Effective and Explainable Crop Field Segmentation},
  ISBN = {9783031918353},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-031-91835-3_8},
  DOI = {10.1007/978-3-031-91835-3_8},
  booktitle = {Computer Vision – ECCV 2024 Workshops},
  publisher = {Springer Nature Switzerland},
  author = {Rege Cambrin,  Daniele and Poeta,  Eleonora and Pastor,  Eliana and Cerquitelli,  Tania and Baralis,  Elena and Garza,  Paolo},
  year = {2025},
  pages = {115–131}
}
```
