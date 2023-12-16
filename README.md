<p align="center">
    <img src="assets/adaptir_logo.png" width="340">
</p>

## AdaptIR: Parameter Efficient Multi-task Adaptation for Pre-trained Image Restoration Models

[[Paper](https://arxiv.org/pdf/2312.08881.pdf)]  [[Suppl]()] [Project Page][Zhihu(知乎)]


[Hang Guo](https://github.com/csguoh), [Tao Dai](https://cstaodai.com/), [Yuanchao Bai](https://scholar.google.com/citations?user=hjYIFZcAAAAJ&hl=zh-CN), Bin Chen, [Shu-Tao Xia](https://scholar.google.com/citations?hl=zh-CN&user=koAXTXgAAAAJ), Zexuan Zhu


> **Abstract:**  Pre-training has shown promising results on various image restoration tasks, which is usually followed by full finetuning for each specific downstream task (e.g., image denoising). However, such full fine-tuning usually suffers from
the problems of heavy computational cost in practice, due to the massive parameters of pre-trained restoration models, thus limiting its real-world applications. Recently, Parameter Efficient Transfer Learning (PETL) offers an efficient alternative solution to full fine-tuning, yet still faces great challenges for pre-trained image restoration models, due to the diversity of different degradations. To address these issues, we propose AdaptIR, a novel parameter efficient transfer learning method for adapting pre-trained restoration models. Specifically, the proposed method consists of a multi-branch inception structure to orthogonally capture local spatial, global spatial, and channel interactions. In this way, it allows powerful representations under a very low parameter budget. Extensive experiments demonstrate that the proposed method can achieve comparable or even better performance than full fine-tuning, while only using 0.6% parameters.


<p align="center">
    <img src="assets/pipeline.png" style="border-radius: 15px">
</p>

⭐If this work is helpful for you, please help star this repo. Thanks!🤗



## 📑 Contents

- [Visual Results](#visual_results)
- [News](#news)
- [TODO](#todo)
- [Results](#results)
- [Citation](#cite)


## <a name="visual_results"></a>:eyes:Visual Results On Different Restoration Tasks
[<img src="assets/imgsli1.png" height="153"/>](https://imgsli.com/MjI1Njk3) [<img src="assets/imgsli7.png" height="153"/>](https://imgsli.com/MjI1NzIx) [<img src="assets/imgsli5.png" height="153"/>](https://imgsli.com/MjI1NzEx) [<img src="assets/imgsli2.png" height="153"/>](https://imgsli.com/MjI1NzAw)

[<img src="assets/imgsli4.png" height="150"/>](https://imgsli.com/MjI1NzAz) [<img src="assets/imgsli3.png" height="150"/>](https://imgsli.com/MjI1NzAx) [<img src="assets/imgsli6.png" height="150"/>](https://imgsli.com/MjI1NzE2)



## <a name="news"></a> 🆕 News

- **2023-12-12:** arXiv paper available.
- **2023-12-16:** This repo is released.



## <a name="todo"></a> ☑️ TODO

- [x] arXiv version
- [ ] Supplementary matrial
- [ ] Project page
- [ ] Release code
- [ ] Pretrained weights
 

## <a name="results"></a> 🥇 Results

We achieve state-of-the-art adaptation performance on various downstream image restoration tasks. Detailed results can be found in the paper.

<details>
<summary>Evaluation on Second-order Degradation (LR4&Noise30) (click to expand)</summary>

<p align="center">
  <img width="900" src="assets/SR&DN.png">
</p>
</details>


<details>
<summary>Evaluation on Classic SR (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/classicSR.png">
</p>
</details>


<details>
<summary>Evaluation on Denoise&DerainL (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/Dn&DRL.png">
</p>
</details>


<details>
<summary>Evaluation on Heavy Rain Streak Removal (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/DRH.png">
</p>
</details>


<details>
<summary>Evaluation on Low-light Image Enhancement (click to expand)</summary>

<p align="center">
  <img width="500" src="assets/low-light.png">
</p>

</details>


<details>
<summary>Evaluation on Model Scalability (click to expand)</summary>

<p align="center">
  <img width="600" src="assets/scalabiltity.png">
</p>

</details>




## <a name="cite"></a> 🥰 Citation

Please cite us if our work is useful for your research.

```

```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is based on [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet), [IPT](https://github.com/huawei-noah/Pretrained-IPT) and [EDT](https://github.com/fenglinglwb/EDT). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at cshguo@gmail.com

