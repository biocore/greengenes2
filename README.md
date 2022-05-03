# Background

The Greengenes2 phylogeny is based on whole genome information from the [Web of Life](https://biocore.github.io/wol/), and revised with high quality full length 16S from the [Living Tree Project](https://imedea.uib-csic.es/mmg/ltp/) and full length 16S extracted from [bacterial operons](https://www.nature.com/articles/s41592-020-01041-y) using [uDance](https://github.com/balabanmetin/uDance). A seed taxonomy is derived using the mappings from the Web of Life to [GTDB](https://gtdb.ecogenomic.org/). This taxonomy is then augmented using information from the Living Tree Project when possible. The augmented taxonomy is decorated onto the backbone using [tax2tree](https://github.com/biocore/tax2tree).

Using this decorated backbone, all public and private 16S V4 ASVs from [Qiita](https://qiita.ucsd.edu/) pulled from [redbiom](https://github.com/biocore/redbiom/) representing hundreds of thousands of samples, as well as full length mitochondrial and chloroplast 16S (sourced from [SILVA](https://www.arb-silva.de/), are then placed using [DEPP](https://github.com/yueyujiang/DEPP). Fragments are resolved. The resulting tree contains > 15,000,000 tips. 

Fragment resolution can result in fragments being placed on the parent edge of a named node. This can occur if the node representing a clade, such as d__Archaea, does not represent sufficient diversity for the input fragments to place. As a result, prior to reading taxonomy off of the tree, each name from the backbone is evaluated for whether its edge to parent has a single or multifurcation of placements. If this occurs, the name is “promoted”. The idea being that fragments off a named edge to its parent are more like the named node than a sibling.

Following this name promotion, the full taxonomy is then read off the tree providing lineage information for each fragment and sequence represented in the tree. This taxonomy information can be utilized within QIIME 2 by cross referencing your input feature set against what’s present in the tree. By doing so, we can obtain taxonomy for both WGS data (if processed by [Woltka](https://github.com/qiyunzhu/woltka) and 16S V4 ASVs. There is an important caveat though: right now, we can only classify based sequences already represented by the tree, so unrepresented V4 ASVs will be unassigned.  

# What is this repository?

This repository contains the methods and detail for performing taxonomy decoration against a backbone. And, following decoration, to establish the release files.  
