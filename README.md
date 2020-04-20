# Reference-less Quality Estimation of Text Simplification Systems

This repository contains the implementation of the analysis and evaluation methods representend in Stodden et al. (2020). 
In this paper, we investigate the relevance of text simplification an text readability features for Czech, German, 
English, Spanish, and Italian text simplification corpora. Our multi-lingual and multi-domain corpus analysis shows 
that the relevance of different features for text simplification is different per corpora, language, and domain.
This implementation is based on the evaluation method used in [Martin et eal. (2018)]((https://www.aclweb.org/anthology/W18-7005)), but it extends it with more features and enables the analysis of other languages than English.


## Getting Started

### Dependencies

* Python 3.6
* Java
* Optional: [QuEst](https://github.com/ghpaetzold/questplusplus) (install in `resources/tools/quest/`)
* Optional: [TERp](https://github.com/snover/terp) (install in `resources/tools/terp/`)

### Installing

```
git clone https://github.com/rstodden/text-simplification-evaluation.git
cd text-simplification-evaluation
pip install -e .
pip install -r requirements.txt
```


## References

If you use this code, please cite 
* for the multi-lingual and cross-domain analysis: R. Stodden, and L. Kallmeyer (2020, to appear). A multi-lingual and cross-domain analysis of features for text simplification. In Proceedings of the Workshop on Tools and Resources to Empower People with REAding DIfficulties (READI), Marseille, France.
* **and**
* for the original evaluation code: L. Martin, S. Humeau, PE. Mazaré, E. De la Clergerie, A. Bordes, B. Sagot, [*Reference-less Quality Estimation of Text Simplification Systems*](https://www.aclweb.org/anthology/W18-7005/). In Proceedings of the 1st Workshop on Automatic Text Adaptation (ATA), Tilburg, the Netherlands.
  


## License

See the [LICENSE](LICENSE) file for more details.
