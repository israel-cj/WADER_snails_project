# WADER_snails_project

## Testing the feasibility of deep learning approaches to enhance marine macroinvertebrate monitoring

## Overview
This repository contains the code and data used for analyzing images of snails using five different computational models. The performance of these models was compared across five frameworks.

## Models Analyzed
1. **CounTR**: A transformer-based model.
   - Repository: [CounTR](https://github.com/Verg-Avesta/CounTR.git)

2. **Segment Anything (SAM)**: A model developed by Meta AI for image segmentation.
   - Repository: [Segment Anything](https://github.com/facebookresearch/segment-anything.git)

3. **Training-free Object Counting with Prompts**: Leverages SAM using a segmentation model that identifies individual objects based on specified prompts including text inputs, boxes, or points.
   - Repository: [Training-free Object Counting with Prompts](https://github.com/shizenglin/training-free-object-counter.git)

4. **Grounding DINO**: An open-set object detector that leverages a transformer-based architecture with grounded pre-training to detect objects based on text inputs.
   - Repository: [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO.git)

5. **DeepDataSpace**: An open-source framework focused on dataset tools for dataset visualization, labeling, and counting.
   - Repository: [DeepDataSpace](https://github.com/IDEA-Research/deepdataspace)

Three of the models (SAM, Grounding DINO, and Training-free Object Counting with Prompts) count objects based on text inputs. In these cases, the prompt 'snails' was used.

## Repository Structure
- **Code and data**: Contains the code for reproducing our figures and data analysis.
- **Figures and statistical analysis**: Contains the final figures and statistical analysis, which were additionally run in jamovi.

## Additional Information
- The images used in the current analysis, which are not included by the size can be requested from Heather Sugden at [heather.sugden@newcastle.ac.uk](mailto:heather.sugden@newcastle.ac.uk).
- The jamovi project (2024). jamovi (Version 2.5) [Computer Software]. Retrieved from [https://www.jamovi.org](https://www.jamovi.org).

## Citation
Please cite the corresponding works for each model and tool used in this analysis.