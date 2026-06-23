---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 2D landmark digitization from images

This tutorial shows how to digitize landmark coordinates from images and turn
them into a multi-specimen dataset ready for shape analysis with ktch. You will
place the 15 landmarks of
[Chitwood & Otoni (2017)](https://doi.org/10.1093/gigascience/giw008) on
*Passiflora* leaf scans, save the result to TPS format, and run a Generalized
Procrustes Analysis (GPA) followed by a PCA of shape variation across
species.

## Prerequisites

```{code-cell} ipython3
# Uncomment if needed
# !pip install ipympl
# !pip install ktch[data,plot]
```

```{code-cell} ipython3
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ktch.datasets import load_image_passiflora_leaves
from ktch.io import read_tps, write_tps
from ktch.landmark import GeneralizedProcrustesAnalysis
```

## Step 1: The Passiflora 15 landmarks

For background on landmark types and digitization approaches, see
{doc}`../../explanation/landmark`.

We use the 15 landmarks of Chitwood & Otoni (2017), which capture the
vascular branching pattern of *Passiflora* leaves:

- Landmarks 1–6: petiolar junction (where the proximal veins and midvein meet at the leaf base)
- Landmarks 7, 15: tips of the proximal veins
- Landmarks 8, 14: proximal sinuses
- Landmarks 9, 13: tips of the distal veins
- Landmarks 10, 12: distal sinuses
- Landmark 11: leaf tip

You will see these 15 points placed on a real leaf at the end of Step 3.

## Step 2: Load the scan images

We use the Passiflora leaf scan dataset bundled with ktch: 25 flatbed scan
images of 10 *Passiflora* species spanning simple elliptical to deeply lobed
leaf forms. Each image contains multiple leaves from one plant individual,
arranged from tip (youngest) to base (oldest). This dataset is
a subset of the *Passiflora* leaf scan data from
[Chitwood & Otoni (2017)](https://doi.org/10.1093/gigascience/giw008)
(data: [Chitwood & Otoni, 2016](https://doi.org/10.5524/100251)).

```{code-cell} ipython3
data = load_image_passiflora_leaves(as_frame=True)

img_by_idx = {idx: img for idx, img in zip(data.meta.index, data.images)}
labels = {
    idx: f"{row.abbreviation} ({row.species})"
    for idx, row in data.meta.iterrows()
}

print(f"# of images: {len(data.images)}")
print(f"Species: {data.meta['species'].nunique()}")
print(data.meta[["abbreviation", "species"]].value_counts().sort_index())
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(8, 12))
for ax, idx in zip(axes.flatten(), list(data.meta.index)[:4]):
    ax.imshow(img_by_idx[idx])
    ax.set_title(labels[idx])
```

We will digitize landmarks on individual leaves. Pick one scan to work with:

```{code-cell} ipython3
image_id = "Pcae1_1_8"
img = img_by_idx[image_id]

fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(img)
ax.set_title(labels[image_id])
```

### Image coordinate system

In image coordinates, the origin is at the top-left corner. The x-axis points
right and the y-axis points down. Landmark coordinates are recorded in this
system.

## Step 3: Digitize landmarks on a leaf

### Place landmarks interactively

Each scan contains multiple leaves. Choose one leaf and place its 15 landmarks
in order following the scheme described in Step 1. Left-click to add a point,
right-click to undo the last point.

:::{note}
This cell needs an interactive matplotlib backend, provided by
[ipympl](https://matplotlib.org/ipympl/). Install it with `pip install ipympl`
in the same environment as your Jupyter kernel, restart the kernel, then run the
cell locally with `%matplotlib widget` in JupyterLab or VS Code. Capturing clicks
needs a live Python kernel, so the cell cannot run in the static documentation;
the two cells that follow render the view and the result instead, and the rest
of the tutorial uses pre-digitized landmarks so every output is reproducible.

If you get `RuntimeError: 'widget' is not a recognised GUI loop or backend name`,
ipympl is missing from the kernel's environment or the kernel was not restarted
after installing it.

For digitizing large datasets, dedicated GUI tools are recommended; see
{doc}`../../explanation/landmark` for an overview.
:::

```{code-cell} ipython3
:tags: [skip-execution]

%matplotlib widget

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.set_title("Click to place landmarks (right-click to undo)")

landmarks = []
scatter = ax.scatter([], [], c="red", s=50, zorder=5)
texts = []


def on_click(event):
    if event.inaxes != ax:
        return
    if event.button == 1:  # left click: add
        landmarks.append([event.xdata, event.ydata])
    elif event.button == 3 and landmarks:  # right click: undo
        landmarks.pop()

    pts = np.array(landmarks) if landmarks else np.empty((0, 2))
    scatter.set_offsets(pts)

    for t in texts:
        t.remove()
    texts.clear()
    for i, (x, y) in enumerate(landmarks):
        texts.append(ax.annotate(str(i + 1), (x, y), fontsize=8,
                                 color="yellow", ha="center", va="bottom"))
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("button_press_event", on_click)
```

```{code-cell} ipython3
:tags: [skip-execution]

landmarks
```

### The digitization view

The widget's appearance is just an ordinary matplotlib figure; only the
click handling needs a kernel. Rendered statically, the canvas you would
interact with locally looks like this:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(img)
ax.set_title("Click to place landmarks (right-click to undo)")
```

### Pre-digitized landmarks

Because clicking needs a live kernel, the rest of this tutorial uses landmarks
digitized for 70 leaves of five species (*P. caerulea*,
*P. cincinnata*, *P. coriacea*, *P. edulis*, *P. gracilis*).
The coordinates are in the same image (pixel) coordinates the widget records,
so they drop straight into `coords_list`.

```{code-cell} ipython3
:tags: [hide-input]

# Pre-digitized landmarks in image (pixel) coordinates, the same form the
# widget records.
_LANDMARKS = np.array([
    [[292.8, 378.8], [291, 378.6], [292.5, 376.2], [296, 374.6], [298.9, 375.8], [297.5, 376.7], [273.9, 380.2], [268.4, 379], [169.3, 328.1], [282, 368.5], [250.3, 200.5], [298.6, 358], [414.1, 304.7], [314.3, 368.7], [310.5, 371.6]],  # Pcae1_1
    [[747.1, 377.6], [743.9, 377.6], [746.6, 374.8], [752.1, 373.4], [756.7, 376.3], [755.2, 377.7], [702.1, 397], [693.6, 394.9], [571.9, 362.2], [731.5, 359.2], [738.2, 141.9], [761.3, 356.4], [923.5, 333.5], [815.5, 383], [806.4, 386]],  # Pcae1_2
    [[442.5, 809.5], [440.4, 808.8], [443.7, 806], [448.7, 804.8], [452.2, 807.1], [451.2, 808.4], [404.6, 830.3], [394.6, 830.3], [258, 800.6], [428.6, 790.4], [416.8, 556.9], [459.3, 781.2], [620.1, 731.9], [517, 811.6], [506.6, 816.1]],  # Pcae1_3
    [[847.5, 810.5], [845, 809.7], [850.4, 808.3], [855.3, 807.9], [858.8, 810.2], [856.7, 811.2], [785.7, 822.1], [777.7, 819.6], [694.4, 770.9], [836.4, 789], [833.5, 578.4], [867.8, 785], [1016.2, 750.4], [938.1, 815.8], [927.2, 821.2]],  # Pcae1_4
    [[1081.2, 592.7], [1079.7, 592.2], [1083.9, 590.8], [1088.3, 589.1], [1090.7, 590.8], [1090.2, 592.2], [1025.7, 599.7], [1017.2, 597.5], [950.5, 560.6], [1072.1, 574.2], [1084.9, 394.5], [1099.2, 569], [1227, 532.6], [1157.8, 588.2], [1150.2, 590.1]],  # Pcae1_5
    [[281.5, 1184.4], [280.2, 1183.7], [282.8, 1182.2], [287.2, 1180.6], [290.8, 1181.6], [290.2, 1182.9], [230.8, 1196.2], [221.3, 1194.6], [138.3, 1151.1], [267.9, 1158.2], [292.4, 965.6], [300, 1160.8], [415.8, 1137.4], [356.6, 1183.6], [350.5, 1188.1]],  # Pcae1_6
    [[634.9, 1114.5], [633.2, 1113.7], [634.1, 1110.3], [640.3, 1109.5], [642.4, 1112.4], [641.5, 1114], [577.6, 1140.9], [566, 1137.9], [487.3, 1099.8], [614.1, 1082.7], [625.8, 875.7], [663, 1085.3], [791.6, 1116.6], [727.7, 1143.9], [716.2, 1144.9]],  # Pcae1_7
    [[1025.9, 1164.5], [1024, 1163.8], [1026.7, 1161.2], [1032, 1161.2], [1034.4, 1163.5], [1033.4, 1165.2], [963.8, 1182], [951.9, 1178.5], [884.8, 1111], [997.4, 1128.9], [1020.5, 919.1], [1060.2, 1130.4], [1185.2, 1123.4], [1140.2, 1177.6], [1128.8, 1182]],  # Pcae1_8
    [[297.5, 459.5], [296.8, 458.5], [297.2, 455.9], [305.3, 456.9], [305.1, 458.3], [304.1, 459.4], [242.6, 453.6], [239, 447.1], [247, 400.1], [249.2, 385.6], [288.1, 207.5], [366.6, 434.6], [363, 442.9], [347.8, 468.6], [343.9, 471.6]],  # Pcae1_9
    [[667, 451.5], [666, 449.9], [668.2, 449.1], [674.7, 448], [675.1, 451.3], [674.8, 453.3], [596, 435.1], [593.7, 427.7], [604.5, 349.9], [609.5, 339.3], [672.5, 196.6], [750.4, 343.8], [749, 368.1], [744.6, 442.8], [739.5, 453.4]],  # Pcae1_10
    [[1034.1, 451.2], [1033.5, 450.5], [1034.8, 446.1], [1040.6, 446.3], [1041.2, 449.2], [1040.5, 450.8], [1006.2, 469], [998.2, 465.8], [962.5, 424.6], [960.5, 412.7], [1049.5, 209.6], [1131.3, 412.5], [1124.8, 428.6], [1090.7, 468.3], [1085.2, 470.4]],  # Pcae1_11
    [[490.1, 883.1], [488.8, 882.4], [489.6, 880.3], [496.3, 878.4], [495.9, 881.9], [495.8, 882.9], [458.9, 899.3], [447.9, 896.9], [428, 873.3], [423.9, 866.1], [507.1, 661.6], [582.1, 839], [578.8, 851.1], [555.9, 888.7], [550.8, 893.4]],  # Pcae1_12
    [[800.4, 926.4], [800, 925], [800.6, 922.9], [806.8, 920], [806.7, 923.8], [807, 925.9], [754.2, 939.2], [745.2, 934.1], [724.7, 900], [721.7, 884.1], [796.6, 729.8], [897.4, 867.7], [894.7, 886.6], [878.6, 922.8], [872.1, 928.5]],  # Pcae1_13
    [[336.4, 1218.4], [336.5, 1217.1], [336.9, 1212.1], [341.6, 1209.6], [341.3, 1217], [341.2, 1218.7], [294.2, 1224.1], [284.2, 1217.3], [268.3, 1153.7], [271.5, 1141.3], [344.9, 1064], [413.9, 1151.4], [414.6, 1165.7], [403.5, 1215.7], [396.1, 1223.4]],  # Pcae1_14
    [[599.1, 1235], [598.7, 1233.6], [599.3, 1229.9], [603.8, 1225.9], [603.9, 1231.8], [603.8, 1233.7], [557.7, 1243.2], [547.4, 1238.1], [530.3, 1184.8], [530.7, 1175.7], [601.8, 1093], [668.4, 1152.3], [672.9, 1163.8], [655.2, 1242.5], [646.2, 1247]],  # Pcae1_15
    [[898.3, 1197.2], [898.3, 1196.2], [897.6, 1189.7], [900.8, 1192.3], [902.1, 1196.7], [902.3, 1198.3], [860.7, 1195.8], [855.9, 1192.1], [851.8, 1150.9], [854.9, 1144.9], [887.2, 1126.8], [941.7, 1161.9], [943.3, 1171.2], [937.2, 1188.8], [933.2, 1196.9]],  # Pcae1_16
    [[350.6, 427.9], [349.1, 425.3], [353.1, 422.1], [359.1, 421.6], [363.5, 424.8], [362.6, 428.1], [229.4, 466.7], [345.1, 423.9], [203.6, 311.5], [353.4, 421.1], [348.6, 195.8], [359.9, 421.8], [512.4, 297.2], [364.5, 424.4], [492.4, 461.6]],  # Pcin1_1
    [[363.3, 855.1], [361.9, 851.5], [364.9, 847.3], [370.6, 845.5], [374.8, 848], [376.2, 850.9], [189.4, 913.2], [353.6, 849.7], [130.4, 729.1], [364, 845.3], [359.1, 553.6], [370.9, 843.8], [556.1, 661.3], [380.9, 843.4], [558.2, 876.4]],  # Pcin1_2
    [[333.4, 1329.4], [334.2, 1326.4], [338.7, 1323.6], [345, 1325.9], [347.8, 1330.9], [346.2, 1334.2], [145.7, 1356.6], [331.2, 1321.5], [135.3, 1127.5], [338.9, 1320.6], [364.7, 981.5], [344.8, 1324.1], [587.3, 1160.8], [353.1, 1328.7], [530.8, 1409.4]],  # Pcin1_3
    [[888.6, 532.4], [888.6, 529.1], [893.1, 526.5], [899.3, 526.1], [903.2, 529.5], [903.1, 533.7], [708.7, 556.4], [876.6, 524.9], [674.8, 322], [892.3, 523.9], [895.5, 179.7], [899.8, 523.5], [1148.7, 337], [910.4, 526.7], [1093.4, 569]],  # Pcin1_4
    [[841.8, 1009.8], [841.2, 1006.3], [846, 1002.7], [853.9, 1002.7], [857.7, 1005.1], [857.3, 1009.3], [642.1, 1042.5], [832.3, 1001.4], [603.1, 808.3], [845.6, 1000.4], [861.2, 633.3], [853.4, 1000.7], [1102.3, 801.1], [876.8, 995.1], [1043.8, 1064.4]],  # Pcin1_5
    [[878.7, 1500.1], [877.2, 1497], [882.9, 1493.1], [891.2, 1491.4], [897.1, 1494], [897.1, 1498.2], [688.8, 1612.2], [833.5, 1495.2], [524.3, 1348.9], [881.9, 1490], [856.3, 1032.4], [890.3, 1487.2], [1210.6, 1262.4], [920.1, 1485.8], [1096.9, 1593.5]],  # Pcin1_6
    [[403.9, 612.5], [403.7, 609.7], [410.1, 604.6], [419.1, 604.3], [424.7, 608.8], [424.1, 613.5], [189.3, 728.4], [350.3, 602.5], [55.8, 384.2], [410.3, 600.8], [418.8, 98.2], [420.3, 601.1], [763.2, 372], [448, 602], [672, 692]],  # Pcin1_7
    [[405.9, 1312.2], [408, 1308.2], [416.4, 1306.5], [425.9, 1306.1], [433.4, 1309.8], [434, 1315.4], [71.4, 1309.1], [352.6, 1282.5], [55.8, 962.8], [415.9, 1299.1], [457.5, 734.9], [428, 1298.9], [868.5, 1049.1], [473, 1302.6], [701, 1509.6]],  # Pcin1_8
    [[607.1, 830.3], [608.4, 826.1], [618.3, 824.2], [628.8, 822.8], [635, 827.5], [635.8, 833.3], [321.9, 914.7], [469.7, 821], [152.2, 527.2], [615.9, 816.7], [593.7, 161.9], [630.8, 811.4], [1071.1, 502], [739.4, 790.8], [988.5, 878]],  # Pcin1_9
    [[824.3, 1504.5], [824.7, 1500.5], [834, 1497.2], [843, 1493.5], [851.3, 1493.5], [854, 1498.2], [538.9, 1657.1], [720.1, 1500.8], [331.5, 1302.1], [828.5, 1487.4], [577.3, 903.7], [844.5, 1478.6], [1183.8, 1099.3], [914.6, 1452.1], [1147, 1520.6]],  # Pcin1_10
    [[426.2, 636.6], [428.5, 633.3], [437.8, 635.2], [446.3, 635], [452.7, 636.1], [455.2, 640.3], [174.9, 677.4], [245.9, 621.2], [87, 365.2], [434.4, 624], [474.1, 109.6], [450.5, 621.8], [804.5, 373.9], [652.3, 621.1], [725.2, 673.5]],  # Pcin1_11
    [[416.3, 1137.9], [415.5, 1134.5], [429.2, 1139.2], [437, 1141.3], [465.2, 1123.8], [462.4, 1126.2], [237.9, 1123.1], [223.5, 1103], [181.8, 975.7], [380, 1051.9], [440.8, 725.1], [485.6, 1050], [675.6, 970.5], [617.8, 1108.6], [610.3, 1123.6]],  # Pcin1_12
    [[761.6, 1387.4], [761.1, 1384.2], [765.9, 1381.3], [771.6, 1381.4], [775.9, 1383.5], [776.9, 1386.5], [636.4, 1366.9], [627.5, 1356.4], [606.2, 1243.7], [714.3, 1290], [747.6, 1079.7], [819.5, 1270.3], [904.7, 1210.7], [900.4, 1328.5], [896.6, 1342.8]],  # Pcin1_13
    [[1044.3, 1506.1], [1045.6, 1503.3], [1053.3, 1505.6], [1059.5, 1505.9], [1067.1, 1508.7], [1063.6, 1510.8], [946.6, 1468.8], [944.4, 1462.4], [947.7, 1420.2], [1036.7, 1464.6], [1077.4, 1344.7], [1083.9, 1480], [1186.7, 1488.4], [1158.2, 1534], [1145.3, 1542.6]],  # Pcin1_14
    [[843.9, 1566.9], [842.3, 1562.7], [846.6, 1558.5], [853.9, 1557.5], [855.8, 1561.8], [853.6, 1565.3], [704.9, 1542.3], [688.8, 1533.7], [661.8, 1507.5], [775.5, 1484.4], [854.4, 1481.1], [916.6, 1493.5], [1036.6, 1524.6], [989.2, 1562.2], [979, 1569.5]],  # Pcor1_1
    [[402.8, 1346], [400.3, 1342.1], [405.1, 1336.7], [411.6, 1336.5], [415.4, 1341.4], [413.2, 1345.4], [220.3, 1352.3], [197.7, 1345.4], [101.3, 1287.6], [269.7, 1201.8], [411.8, 1188.1], [617.3, 1223.9], [730.1, 1290.4], [645.2, 1348.2], [618.2, 1362.6]],  # Pcor1_2
    [[767.9, 972.4], [765.4, 969.8], [770.6, 966.5], [775.9, 966.5], [779.8, 969.8], [777.5, 974], [573.3, 1009.3], [550.6, 1001.8], [450.6, 941.6], [628.7, 830.1], [771.5, 798.1], [937.1, 837.8], [1078.6, 937.1], [989.5, 988.7], [956.3, 1003.2]],  # Pcor1_3
    [[789.5, 567.8], [787.8, 564.6], [792, 562.3], [796.8, 562.7], [803.8, 565.8], [802.1, 569], [607.9, 578.4], [587.6, 570.2], [494.8, 517.1], [654.9, 433.2], [797.8, 393.5], [950.6, 434.6], [1119.8, 531.7], [1029.2, 584.2], [990.3, 597.8]],  # Pcor1_4
    [[324.9, 713.7], [324.2, 710.9], [326.5, 707.8], [334.2, 708.3], [337.5, 711.1], [336.1, 713.9], [161.5, 723.5], [141.1, 706.9], [46.2, 626.1], [194.6, 562.9], [335.6, 527.4], [460.7, 574.9], [624.2, 662.3], [527.1, 718.1], [493.9, 734]],  # Pcor1_5
    [[291.3, 408.7], [291, 406.4], [293.3, 403.6], [298.5, 402.7], [302.9, 405.4], [302.2, 408.3], [177, 393.1], [168.6, 381.6], [112.2, 313.2], [222.3, 282.7], [308.8, 220.2], [372.7, 283.7], [490, 315.5], [426.4, 383.5], [406.4, 402.7]],  # Pcor1_6
    [[610.8, 270.8], [609.3, 269.9], [611.2, 266.7], [615.2, 264.8], [619.7, 266.2], [619.2, 268.1], [562.6, 286.8], [556.3, 282.1], [538.5, 220.9], [542.3, 205.2], [596.6, 114.5], [653.4, 172.2], [663.9, 188.1], [677.5, 218.5], [678.4, 231.1]],  # Pcor1_7
    [[271.2, 334.2], [269.2, 332.1], [298.1, 359.4], [305.6, 358.7], [315.6, 350.6], [314.2, 352], [224.8, 324.8], [214.7, 318.3], [180.7, 246.1], [278.3, 312.7], [275.5, 110.1], [319.3, 311.7], [427.8, 243.3], [340.5, 348.7], [335.6, 352.7]],  # Pedu1_1
    [[363.4, 760.2], [361.4, 757.9], [372.7, 759.5], [380.3, 758.1], [390.8, 753.9], [388.6, 757.2], [312.7, 764.2], [290, 756.7], [138.8, 562.2], [315.2, 663.7], [386.2, 360.3], [412.7, 666.9], [574, 531.2], [482.3, 727.2], [466.5, 737.1]],  # Pedu1_2
    [[407.2, 1429.2], [405.9, 1427.1], [422, 1444.9], [432.1, 1443.5], [455.2, 1436.2], [449.6, 1440.2], [318.5, 1409.1], [288.2, 1394.7], [176.6, 1081.5], [367.3, 1250.5], [472.1, 826.8], [530.9, 1289.7], [805.9, 1165.9], [625.6, 1416.1], [599.4, 1431.3]],  # Pedu1_3
    [[966.2, 1046.5], [961.7, 1043.7], [991.3, 1047.6], [1004.4, 1043.2], [1020.1, 999.2], [1018.3, 1004.9], [736.2, 1064.9], [701.7, 1049.7], [493.6, 779.1], [805.1, 860.4], [754.7, 375.3], [1004.3, 777], [1163.4, 556.4], [1155.9, 869.6], [1136.1, 902.7]],  # Pedu1_4
    [[428.6, 800.6], [427.3, 797.6], [452.5, 832.4], [463.6, 834], [512.3, 813.2], [508.1, 816.3], [314.8, 769.5], [291.4, 746.6], [204.4, 313.2], [333, 458.7], [546.4, 70.6], [633.1, 556], [883.2, 408.7], [717.6, 766.8], [702.9, 784.9]],  # Pedu1_5
    [[561.3, 1386.5], [561.2, 1383.7], [574, 1456.8], [589.4, 1461.7], [659, 1450.1], [653.3, 1453.3], [457, 1294.4], [452.2, 1262.9], [714.4, 860.7], [760.7, 825.6], [1127.3, 624.7], [946.7, 1205.9], [1255.3, 1220.8], [924.2, 1510.1], [888.4, 1530]],  # Pedu1_6
    [[247.4, 1428.3], [244.5, 1421.6], [280.1, 1488.6], [291.6, 1488.6], [351.6, 1405.4], [348.3, 1409.6], [123.2, 1375.9], [98.3, 1338.4], [131.7, 529.8], [150.4, 474.2], [325.5, 101.2], [486.3, 646.8], [498.1, 725.4], [469.4, 1362.2], [454.1, 1401.4]],  # Pedu1_7
    [[766, 1534.2], [764.6, 1529.2], [796.6, 1626.2], [809.6, 1629.3], [858.4, 1569], [855.1, 1574.6], [666, 1489.1], [641, 1451.7], [642.6, 777], [650.4, 679.6], [829.7, 65.6], [1043.7, 641.7], [1057.6, 751.5], [1009.1, 1513.1], [982.8, 1562.7]],  # Pedu1_8
    [[175.5, 752.3], [175.9, 748.3], [181.3, 819.1], [195.3, 834.3], [248.7, 854.4], [242.2, 853.9], [93.2, 588], [108.2, 531.7], [686.4, 154.9], [779.6, 123.9], [1116.3, 132.8], [1009.1, 578.9], [957.7, 642.7], [384.1, 978.2], [333.9, 962.3]],  # Pedu1_9
    [[355, 1314.5], [356.8, 1312.2], [317.1, 1412.2], [327.3, 1427.2], [442.2, 1461.7], [436.2, 1461.3], [305.6, 1203.7], [326.6, 1154.6], [762.8, 925.2], [807.6, 926.8], [1151.9, 962.4], [1034.2, 1319.7], [972.8, 1386.7], [641.1, 1583.7], [600.8, 1583.5]],  # Pedu1_10
    [[152, 470.9], [153.3, 467.6], [143.2, 522.8], [149.5, 537.5], [200.1, 567.6], [196.3, 567.1], [115.6, 361.7], [124, 333.7], [478.5, 84.4], [534.1, 74.8], [897, 95.3], [743.6, 476.9], [715.7, 512.7], [267.3, 670.9], [233.6, 653.6]],  # Pedu1_11
    [[161, 1150], [164.2, 1147.4], [132.5, 1203], [141.5, 1215.2], [195.7, 1242.8], [192.5, 1242.3], [160.9, 1043.7], [174.4, 1025.7], [468.4, 899], [519.9, 892], [876.2, 939.4], [635.1, 1244.4], [572.2, 1282.5], [251.7, 1323.8], [235.2, 1314.7]],  # Pedu1_12
    [[794.3, 701.1], [796.7, 699.4], [778.2, 743.8], [782.9, 753.4], [819.4, 773.3], [816.3, 772.4], [787.7, 649.4], [797.8, 635], [996.3, 488.2], [1030.4, 482.3], [1267, 520.2], [1193.8, 698.3], [1155, 738.9], [900, 837.1], [869.1, 827.9]],  # Pedu1_13
    [[829.9, 1214.7], [831.2, 1212.2], [822.4, 1245.4], [828.6, 1253.1], [887.5, 1272.2], [884.6, 1272.4], [820.1, 1173.6], [826.5, 1165.4], [1039.5, 1019.4], [1067.3, 1013.8], [1233.1, 1020], [1150.2, 1202.4], [1136.6, 1228.3], [964.4, 1309.3], [936.7, 1312.4]],  # Pedu1_14
    [[859.3, 1486.3], [861.1, 1483.9], [863.3, 1493.1], [863.6, 1501], [888.4, 1515], [884.8, 1514.1], [855.9, 1446.6], [860.5, 1433], [951.5, 1368.5], [973.7, 1361.2], [1139, 1369.4], [1083.4, 1480], [1069.4, 1496.5], [925.9, 1538.9], [914.7, 1537.2]],  # Pedu1_15
    [[244.3, 364.6], [242.8, 363.6], [253, 360.8], [257.7, 360.6], [264.4, 363.6], [264.9, 365.2], [229, 366.9], [224.3, 365.7], [183.2, 343.8], [222.4, 331.8], [254, 266.2], [275.4, 331.9], [326.3, 346.6], [290.6, 363.6], [286.3, 365]],  # Pgra1_1
    [[576.1, 434.6], [574.4, 433.2], [581.8, 433.7], [587.1, 434], [597, 433], [594.9, 433.9], [543.8, 449.8], [529.5, 447], [496.9, 386.3], [540.7, 365.5], [585.3, 306.8], [628.9, 368.8], [676.1, 383.9], [642.5, 444.3], [631.5, 449.4]],  # Pgra1_2
    [[853.6, 291.2], [852.3, 289.5], [854.3, 285.4], [860, 285.3], [867.7, 287.5], [865.7, 288.6], [809.1, 305.7], [799.1, 304.7], [736.6, 209.9], [798, 193.7], [850.6, 112.9], [918.2, 187.6], [978.4, 203.3], [922.7, 301.7], [909.7, 307.1]],  # Pgra1_3
    [[984.6, 657.9], [981.9, 656.9], [996.1, 655.3], [1002.1, 654.8], [1014.5, 654.8], [1011.7, 655.7], [910.6, 680.5], [896.3, 673.3], [846.8, 569.3], [911.8, 545], [979.4, 450.6], [1067.3, 534], [1141.3, 555], [1098.6, 660.7], [1083.9, 670.5]],  # Pgra1_4
    [[265.9, 887.5], [263.1, 886.8], [279.8, 886.6], [286.2, 886.6], [311, 881.7], [308.6, 883.3], [194.3, 910], [172.7, 900.2], [129.1, 772.6], [195.7, 768.9], [277.4, 665.1], [357.2, 759.3], [441.4, 781.5], [426.3, 867.4], [412.7, 882.9]],  # Pgra1_5
    [[691.1, 1112.1], [687.3, 1110.7], [701, 1108.9], [708.7, 1108.9], [725.2, 1106.7], [723.1, 1108.6], [602, 1142.3], [584.2, 1137.8], [514.9, 1013.8], [604.5, 976.4], [686.9, 880.8], [792.1, 958.1], [876.1, 986.9], [829.1, 1112.4], [811.3, 1123.4]],  # Pgra1_6
    [[318.8, 1315.4], [315.1, 1313.6], [332.5, 1313.8], [340.3, 1313.6], [362.8, 1309.3], [360.2, 1310.8], [239.9, 1348.1], [222.7, 1344.7], [153.6, 1206.3], [224.7, 1172.2], [326.6, 1082.2], [438.6, 1164.3], [512.4, 1195.6], [487.3, 1315.4], [467.5, 1330.1]],  # Pgra1_7
    [[935, 1535.3], [932.1, 1533.5], [957.7, 1538.2], [964.3, 1538.2], [989.5, 1537.7], [984.9, 1539.5], [817.2, 1558.5], [806.6, 1549.6], [781, 1426.2], [851.5, 1389], [960, 1296.5], [1070.6, 1389.8], [1147.5, 1433], [1095.7, 1568.8], [1074.8, 1580.9]],  # Pgra1_8
    [[261, 453.4], [258.4, 451.9], [290.9, 455.5], [296.7, 456.4], [323, 457.5], [318.6, 458.3], [145.6, 486.3], [129.8, 475.8], [98.8, 345.8], [156.7, 309.2], [293, 200.7], [425.7, 301], [495.3, 353.1], [441.5, 496.4], [412.2, 503.1]],  # Pgra1_9
    [[827, 572.5], [824.7, 570.4], [852, 571.6], [859.2, 571.1], [898, 564.6], [894.1, 566.5], [711, 619.1], [695.9, 612.3], [624.7, 423.4], [729.6, 369.5], [861.4, 268.7], [1009.2, 372], [1081.6, 422.7], [1059.4, 571.1], [1036.5, 595.4]],  # Pgra1_10
    [[276, 1186.7], [272.1, 1186.2], [298.7, 1186.5], [305.6, 1186.2], [330.3, 1186.2], [325.7, 1187.9], [145.3, 1239.7], [120.5, 1234.3], [54.5, 1024], [129.3, 981.5], [276.8, 851.5], [456.9, 949.9], [539.3, 982.6], [478.8, 1207.3], [461.6, 1221]],  # Pgra1_11
    [[806.4, 1348.9], [803.9, 1346.5], [848.5, 1359.2], [855, 1358.4], [875.3, 1357.8], [871.2, 1360.1], [653.7, 1370.8], [631.6, 1359.8], [612.2, 1171.3], [700.4, 1128.5], [843, 1028.9], [994.9, 1124.3], [1074.9, 1168.7], [1024.2, 1381.8], [993.1, 1392.8]],  # Pgra1_12
    [[364.4, 609.7], [362, 607.8], [400.2, 615.8], [407.2, 614.6], [436.1, 610.2], [432.1, 612], [228.2, 635.6], [202.4, 618.4], [175, 425.1], [237.9, 381.4], [396.6, 286.3], [555.8, 378.4], [626.4, 422.5], [580.9, 632.6], [541.1, 644.5]],  # Pgra1_13
    [[917.3, 622.8], [915, 621.1], [944.8, 630.9], [951, 629.1], [976.2, 627], [971.8, 628.8], [822.6, 642.9], [797.2, 635.6], [760.7, 433.3], [814.2, 400.1], [955.2, 324.8], [1103, 427.9], [1149.1, 466.2], [1101, 658.1], [1070.1, 664.8]],  # Pgra1_14
    [[357.2, 1159.8], [355.2, 1157.5], [387.6, 1179.2], [393.1, 1179.6], [415, 1170.8], [412.9, 1172.7], [264.1, 1171.9], [249.9, 1163.6], [230.6, 1027.3], [259.1, 1002], [392.4, 925], [519.6, 992.5], [566, 1021.2], [514.8, 1178.2], [491.6, 1185]],  # Pgra1_15
    [[800.1, 1021], [798.8, 1019.3], [801.9, 1017.5], [805.2, 1016.5], [807.2, 1018.9], [807.2, 1021.5], [738.1, 1025], [721.1, 1020.3], [687.3, 911.1], [709.6, 889.2], [796.2, 832.9], [888, 885.7], [913.3, 907.7], [859.7, 1022.1], [839.7, 1024.3]],  # Pgra1_16
    [[689.5, 1305.4], [686.6, 1303.7], [704, 1312.9], [709.8, 1311.7], [726.4, 1304], [724.4, 1305.9], [613.1, 1299.8], [605.9, 1291.8], [606.5, 1215.2], [642.5, 1199.7], [704, 1153.3], [777.2, 1199.5], [805.2, 1208.6], [799, 1290.4], [789.1, 1299.6]],  # Pgra1_17
    [[1002.4, 1397], [1002.6, 1394.9], [1003.4, 1387.7], [1007.3, 1386], [1007.9, 1393.9], [1007.9, 1396.3], [932.7, 1359.6], [926.9, 1348.6], [929.6, 1271.3], [939.7, 1258.9], [1008.9, 1234.4], [1086.7, 1277.6], [1092.4, 1300.2], [1075.5, 1360.3], [1070.3, 1370.6]],  # Pgra1_18
])

_LEAF_IDS = [
    "Pcae1_1", "Pcae1_2", "Pcae1_3", "Pcae1_4", "Pcae1_5", "Pcae1_6", "Pcae1_7", "Pcae1_8",
    "Pcae1_9", "Pcae1_10", "Pcae1_11", "Pcae1_12", "Pcae1_13", "Pcae1_14", "Pcae1_15",
    "Pcae1_16", "Pcin1_1", "Pcin1_2", "Pcin1_3", "Pcin1_4", "Pcin1_5", "Pcin1_6",
    "Pcin1_7", "Pcin1_8", "Pcin1_9", "Pcin1_10", "Pcin1_11", "Pcin1_12", "Pcin1_13",
    "Pcin1_14", "Pcor1_1", "Pcor1_2", "Pcor1_3", "Pcor1_4", "Pcor1_5", "Pcor1_6",
    "Pcor1_7", "Pedu1_1", "Pedu1_2", "Pedu1_3", "Pedu1_4", "Pedu1_5", "Pedu1_6", "Pedu1_7",
    "Pedu1_8", "Pedu1_9", "Pedu1_10", "Pedu1_11", "Pedu1_12", "Pedu1_13", "Pedu1_14",
    "Pedu1_15", "Pgra1_1", "Pgra1_2", "Pgra1_3", "Pgra1_4", "Pgra1_5", "Pgra1_6",
    "Pgra1_7", "Pgra1_8", "Pgra1_9", "Pgra1_10", "Pgra1_11", "Pgra1_12", "Pgra1_13",
    "Pgra1_14", "Pgra1_15", "Pgra1_16", "Pgra1_17", "Pgra1_18"
]
_SPECIES = [
    "caerulea", "caerulea", "caerulea", "caerulea", "caerulea", "caerulea", "caerulea",
    "caerulea", "caerulea", "caerulea", "caerulea", "caerulea", "caerulea", "caerulea",
    "caerulea", "caerulea", "cincinnata", "cincinnata", "cincinnata", "cincinnata",
    "cincinnata", "cincinnata", "cincinnata", "cincinnata", "cincinnata", "cincinnata",
    "cincinnata", "cincinnata", "cincinnata", "cincinnata", "coriacea", "coriacea",
    "coriacea", "coriacea", "coriacea", "coriacea", "coriacea", "edulis", "edulis",
    "edulis", "edulis", "edulis", "edulis", "edulis", "edulis", "edulis", "edulis",
    "edulis", "edulis", "edulis", "edulis", "edulis", "gracilis", "gracilis", "gracilis",
    "gracilis", "gracilis", "gracilis", "gracilis", "gracilis", "gracilis", "gracilis",
    "gracilis", "gracilis", "gracilis", "gracilis", "gracilis", "gracilis", "gracilis",
    "gracilis"
]
_IMAGE_IDS = [
    "Pcae1_1_8", "Pcae1_1_8", "Pcae1_1_8", "Pcae1_1_8", "Pcae1_1_8", "Pcae1_1_8",
    "Pcae1_1_8", "Pcae1_1_8", "Pcae1_9_16", "Pcae1_9_16", "Pcae1_9_16", "Pcae1_9_16",
    "Pcae1_9_16", "Pcae1_9_16", "Pcae1_9_16", "Pcae1_9_16", "Pcin1_1_6", "Pcin1_1_6",
    "Pcin1_1_6", "Pcin1_1_6", "Pcin1_1_6", "Pcin1_1_6", "Pcin1_7_8", "Pcin1_7_8",
    "Pcin1_9_10", "Pcin1_9_10", "Pcin1_11_14", "Pcin1_11_14", "Pcin1_11_14", "Pcin1_11_14",
    "Pcor1_1_7", "Pcor1_1_7", "Pcor1_1_7", "Pcor1_1_7", "Pcor1_1_7", "Pcor1_1_7",
    "Pcor1_1_7", "Pedu1_1_4", "Pedu1_1_4", "Pedu1_1_4", "Pedu1_1_4", "Pedu1_5_6",
    "Pedu1_5_6", "Pedu1_7_8", "Pedu1_7_8", "Pedu1_9_10", "Pedu1_9_10", "Pedu1_11_15",
    "Pedu1_11_15", "Pedu1_11_15", "Pedu1_11_15", "Pedu1_11_15", "Pgra1_1_8", "Pgra1_1_8",
    "Pgra1_1_8", "Pgra1_1_8", "Pgra1_1_8", "Pgra1_1_8", "Pgra1_1_8", "Pgra1_1_8",
    "Pgra1_9_12", "Pgra1_9_12", "Pgra1_9_12", "Pgra1_9_12", "Pgra1_13_18", "Pgra1_13_18",
    "Pgra1_13_18", "Pgra1_13_18", "Pgra1_13_18", "Pgra1_13_18"
]

coords_list = list(_LANDMARKS)
```

### After placing the 15 landmarks

This is what one digitized leaf looks like;
the 15 landmarks of the first leaf of the chosen scan:

```{code-cell} ipython3
demo_idx = 0
demo_pts = coords_list[demo_idx]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.plot(demo_pts[:, 0], demo_pts[:, 1], "o", color="red", markersize=6)
for i, (x, y) in enumerate(demo_pts):
    ax.annotate(str(i + 1), (x, y), color="yellow", fontsize=12,
                ha="center", va="bottom")

# Zoom to the digitized leaf for legibility.
x0, y0 = demo_pts.min(0)
x1, y1 = demo_pts.max(0)
mx, my = 0.35 * (x1 - x0), 0.35 * (y1 - y0)
ax.set_xlim(x0 - mx, x1 + mx)
ax.set_ylim(y1 + my, y0 - my)
ax.set_title(f"{_LEAF_IDS[demo_idx]} — {labels[image_id]}")
```

## Step 4: Save the landmarks to TPS

TPS is the standard landmark file format (see {doc}`../../explanation/io` and {doc}`../../how-to/data/load_tps` for the details).
`write_tps` stores the coordinates together with per-specimen metadata.

```{code-cell} ipython3
meta = pd.DataFrame(
    {"species": _SPECIES, "image_id": _IMAGE_IDS},
    index=pd.Index(_LEAF_IDS, name="leaf_id"),
)
meta.head()
```

```{code-cell} ipython3
tps_path = Path(tempfile.gettempdir()) / "passiflora_landmarks.tps"

write_tps(
    tps_path,
    coords_list,
    idx=meta.index.tolist(),
    image_path=[f"{idx}.png" for idx in meta["image_id"]],
    comments=meta["species"].tolist(),
)
```

## Step 5: Aggregate into a multi-specimen dataset

Reading the file back gives one record per specimen, from which we recover both
the coordinates and the metadata we stored:

```{code-cell} ipython3
specimens = read_tps(tps_path)

coords_list = [t.to_numpy() for t in specimens]
leaf_ids = [t.specimen_name for t in specimens]
species = [t.comments for t in specimens]
image_ids = [Path(t.image_path).stem for t in specimens]

print(f"Specimens: {len(coords_list)}, landmarks each: {coords_list[0].shape[0]}")
pd.Series(species, name="count").value_counts().sort_index()
```

The coordinates can also be read directly as a tidy DataFrame with a
`(specimen_id, coord_id)` index:

```{code-cell} ipython3
read_tps(tps_path, as_frame=True).head()
```

### Verify landmarks on the source images

Overlay one leaf per species on its scan to confirm the landmarks fall in the
right place:

```{code-cell} ipython3
:tags: [hide-input]

def plot_leaf(ax, leaf_img, pts, title="", margin=0.35):
    """Show a leaf image with numbered landmarks, zoomed to the configuration."""
    ax.imshow(leaf_img)
    ax.plot(pts[:, 0], pts[:, 1], "o", color="red", markersize=4)
    for i, (x, y) in enumerate(pts):
        ax.annotate(str(i + 1), (x, y), color="yellow", fontsize=7,
                    ha="center", va="bottom")
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    ax.set_xlim(x0 - mx, x1 + mx)
    ax.set_ylim(y1 + my, y0 - my)
    ax.set_title(title)
```

```{code-cell} ipython3
# Index of the first leaf of each species.
first_of_species = {}
for k, sp in enumerate(species):
    first_of_species.setdefault(sp, k)

fig, axes = plt.subplots(2, 3, figsize=(12, 9))
axes = axes.flatten()
for ax, (sp, k) in zip(axes, first_of_species.items()):
    plot_leaf(ax, img_by_idx[image_ids[k]], coords_list[k], title=f"{sp}\n{leaf_ids[k]}")

for ax in axes[len(first_of_species):]:
    ax.axis("off")
```

For *P. gracilis*, the first leaf was treated as deformed,
and the landmarks were placed based on its assumed pre-deformation shape.

## Step 6: Prepare for shape analysis

Generalized Procrustes Analysis removes differences in position, scale, and
rotation, leaving only shape. It expects each specimen as a flat vector of
coordinates in `(x, y)` order, so we stack the landmark arrays into a 2-D array:

```{code-cell} ipython3
X = np.stack([lm.reshape(-1) for lm in coords_list])  # (n_specimens, n_landmarks * 2)

gpa = GeneralizedProcrustesAnalysis()
shapes = gpa.fit_transform(X)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Before GPA: raw configurations, centered for comparison.
for lm in coords_list:
    c = lm - lm.mean(0)
    axes[0].plot(c[:, 0], -c[:, 1], ".", alpha=0.3, markersize=3)
axes[0].set_aspect("equal")
axes[0].set_title("Before GPA (centered raw coordinates)")

# After GPA: Procrustes-aligned shapes.
for row in shapes:
    s = row.reshape(-1, 2)
    axes[1].plot(s[:, 0], -s[:, 1], ".", alpha=0.3, markersize=3)
axes[1].set_aspect("equal")
axes[1].set_title("After GPA (aligned shapes)")
```

PCA of the aligned shapes shows how the five species are distributed in
empirical morphospace:

```{code-cell} ipython3
pca = PCA(n_components=2)
scores = pca.fit_transform(shapes)
evr = pca.explained_variance_ratio_ * 100

species_arr = np.array(species)
fig, ax = plt.subplots(figsize=(7, 6))
for sp, color in zip(sorted(set(species)), plt.cm.tab10.colors):
    m = species_arr == sp
    ax.scatter(scores[m, 0], scores[m, 1], label=sp, color=color,
               s=30, alpha=0.8)
ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
ax.legend(title="species")
ax.set_title("Leaf shape variation (PCA of Procrustes-aligned landmarks)")
```

## Summary

This tutorial walked through a landmarking workflow: placing the 15
Passiflora landmarks on a scan, saving them to TPS together with their metadata,
reading them back into a multi-specimen dataset, and running GPA followed by PCA.
The digitized landmarks are ready for shape analysis with ktch.

## Next steps

- {doc}`../landmark/generalized_Procrustes_analysis` - GPA with PCA in depth
- {doc}`../landmark/gpa_with_semilandmarks` - combining landmarks with semilandmarks
- {doc}`../../how-to/data/load_tps` - reading and writing TPS files
- {doc}`../../explanation/landmark` - landmark theory and digitization approaches
