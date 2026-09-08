# Third-party data and mesh attribution

The separately distributed prepared simulation case combines and adapts material from the sources
below. During anonymous peer review, the bundle is made available only for evaluating the
associated manuscript under the review-only notice included in the archive. No general reuse,
redistribution, sublicensing, or commercial-use rights are granted by the Anonymous Authors.
Third-party terms continue to apply and take precedence where required.

## Heart geometry

**Source:** *Virtual cohort of 1000 synthetic heart meshes from adult human healthy population*,
version 1.0.0.

**Creators:** Cristobal Rodero, Marina Strocchi, Maciej Marciniak, Stefano Longobardi, John
Whitaker, Mark D. O'Neill, Karli Gillette, Christoph M. Augustin, Gernot Plank, Edward J. Vigmond,
Pablo Lamata, and Steven A. Niederer.

**DOI:** https://doi.org/10.5281/zenodo.4506930

## Torso geometry

**Source:** *Dataset with segmentations of 117 important anatomical structures in 1228 CT
images*, version 2.0.1.

**Creator:** Jakob Wasserthal.

**DOI:** https://doi.org/10.5281/zenodo.10047292

The licensed TotalSegmentator `heartchambers_highres` task was used as an intermediate step when
constructing the combined heart--torso mesh. This task is available under separate non-commercial
terms and is not one of TotalSegmentator's openly available tasks:

https://github.com/wasserth/TotalSegmentator#subtasks

## Changes made

The public prepared case is not an unchanged copy of either source. The heart ventricles were
extracted and annotated with simulation surfaces, rule-based fibers, ventricular coordinates, a
fast-endocardial layer, APD-region tags, and stimulus node sets. The selected heart was aligned to
the selected torso anatomy. Heart, blood-pool, basal-cap, lung, and torso regions were then
combined into one tetrahedral openCARP mesh. Files and identifiers were renamed for the anonymous
review artifact.

These adaptations do not imply endorsement by the original creators. The repository's software
and model checkpoints are covered separately by the repository-level review license.
