---
tags:
  - spatialvox
  - flowchart
---
# SpatialVox — model flowchart

One relational forward, `StageB.forward(image, direction_ids, name_ids)`. The frozen Stage A is drawn open, as a subgraph of the Stage B forward it lives inside. Shapes are for the MRI corpus (128³ at 1.25 mm, 23 structures) with batch size `B`. The prose behind every box is in [[SpatialVox]]. An editable version of the same chart is `Flowchart.drawio`, which opens in diagrams.net or the draw.io extension for VS Code and Cursor. One note per module is in `Model Info/` ([[MODEL PHASE A]], [[MODEL PHASE B]]).

```mermaid
flowchart TB
    IMG["image [B,1,128³] float32<br/>MRI, z-scored over the brain"]
    NID["name_ids [B,3] int64<br/>the three anchor names"]
    DID["direction_ids [B,3] int64<br/>one direction per clause"]
    CACHE["optional anchors= [B,3,128³]<br/>precomputed cache, or oracle masks"]
    SWAP["optional boundary_image [B,1,128³]<br/>image-replacement test only"]

    subgraph SA["STAGE A - promptable segmenter - frozen, 17,004,292 params, eval mode, no_grad"]
        direction TB
        ENC["Encoder, 5 stages of ConvBlock + ResBlock, ReLU<br/>32@128³ / 64@64³ / 128@32³ / 256@16³ / 256@8³"]
        VAL["flatten the 8³ bottleneck<br/>values [B,512,256]"]
        POS["PosEnc3D<br/>pos_z + pos_y + pos_x, 6,144 params"]
        KEY["keys = values + pos<br/>[B,512,256]"]
        NP["NamePrompt<br/>Embedding 23 x 256, then Linear 256 to 256"]
        QRY["queries [B,3,256]<br/>one token per anchor name"]
        MHA["MultiheadAttention, 256 dims, 4 heads<br/>Q = names, K = keys, V = values"]
        LN["LayerNorm of attended + queries<br/>aligned queries [B,3,256]"]
        DEC["Decoder, 4 stages<br/>upsample, concat the skip, ConvBlock<br/>256@16³ / 128@32³ / 64@64³ / 32@128³"]
        MH["MaskHead on the finest stage<br/>scaled dot product + per-query bias<br/>logits [B,3,128³]"]
        SIG["sigmoid in float32, then detach"]
        ENC --> VAL --> KEY
        POS --> KEY
        NP --> QRY --> MHA
        KEY --> MHA
        VAL --> MHA
        MHA --> LN --> MH
        ENC --> DEC --> MH --> SIG
    end
    IMG --> ENC
    NID --> NP

    ANCH["A_0, A_1, A_2 - soft anchor masks [B,3,128³]<br/>detached probabilities, not thresholds<br/>NAMES STOP HERE"]
    SIG --> ANCH
    CACHE -.->|"replaces Stage A's output"| ANCH

    subgraph MAP["PositionalMapper3D - the WHERE - 0 params, run under no_grad"]
        direction TB
        CEN["soft_centroids<br/>c_i [B,3,3] world mm, mass_i [B,3]"]
        MAR["margins [B,3,128³] world mm<br/>how far the clause's axis leads the other two"]
        FLD["F_i = sigmoid of margin / tau, tau = 0.5 mm<br/>zeroed if mass_i below min_mass = 1e-6<br/>[B,3,128³]"]
        PRD["where_raw = F_0 · F_1 · F_2<br/>[B,1,128³], never renormalised"]
        WMS["where_mass = mean of where_raw<br/>[B,1]"]
        CEN --> MAR --> FLD --> PRD --> WMS
    end
    ANCH --> CEN
    DID --> MAR

    NULL["NullHead - 1,249 params<br/>log10 of where_mass and 3 masses<br/>MLP 4 / 32 / 32 / 1, no pixels"]
    WMS --> NULL
    CEN -->|"mass_i"| NULL

    subgraph BE["BoundaryEncoder B(I) - the WHAT - 228,528 params - sees only the image"]
        direction TB
        D0["ConvBlock 1 to 16 at 128³, no ResBlock"]
        D1["ConvBlock 16 to 32, stride 2, + ResBlock at 64³"]
        D2["ConvBlock 32 to 32, stride 2, + ResBlock at 32³"]
        U0["upsample, concat skip, ConvBlock 64 to 32 at 64³"]
        U1["upsample, concat skip, ConvBlock 48 to 16 at 128³"]
        D0 --> D1 --> D2 --> U0 --> U1
        D1 -.->|"skip"| U0
        D0 -.->|"skip"| U1
    end
    IMG --> D0
    SWAP -.->|"replaces the image B reads"| D0
    BI["B(I) [B,16,128³]"]
    U1 --> BI

    RES["rescale onto -1..0<br/>log where_raw / 20.7<br/>log where_mass / 20.7, broadcast"]
    PRD --> RES
    WMS --> RES
    CAT["concat [B,25,128³]<br/>B(I) 16 + A 3 + F 3 + where_raw 1 + log where 1 + log mass 1<br/>22 channels if carver_sees_anchors is off, 9 if prompt-only"]
    BI --> CAT
    ANCH -->|"if carver_sees_anchors"| CAT
    FLD --> CAT
    PRD --> CAT
    RES --> CAT

    subgraph CV["Carver - 38,498 params"]
        direction TB
        STEM["stem: ConvBlock 25 to 16, stride 2<br/>[B,16,64³]"]
        BLK["2 x ResBlock 16<br/>[B,16,64³]"]
        UP["head, feature half: 1x1 at 64³<br/>then trilinear upsample of ONE channel<br/>[B,1,128³]"]
        SKIP["plus head, B(I) half: 1x1 on B(I)<br/>full_resolution_skip<br/>[B,1,128³]"]
        HEAD["logits: the one 1x1 head, in two exact halves<br/>zero weight, bias = logit of 0.0016<br/>[B,1,128³]"]
        HM["heatmap: separate 1x1 conv<br/>[B,1,64³]"]
        STEM --> BLK --> UP --> SKIP --> HEAD
        BLK --> HM
    end
    CAT --> STEM
    BI -->|"full-resolution skip"| SKIP

    ALPHA["ablation only, off by default<br/>+ alpha · logit of where_raw"]
    EXC["anchor exclusion<br/>logit = -10 wherever max_i A_i is above 0.5"]
    HEAD --> EXC
    ANCH --> EXC
    PRD -.-> ALPHA -.-> EXC
    SAM["soft_argmax<br/>expectation under a softmax over 64³ cells"]
    HM --> SAM

    LOGITS["logits [B,1,128³]<br/>the target mask"]
    CENTROID["centroid [B,3]<br/>world mm, from the heatmap"]
    VALID["valid [B]<br/>one logit: do the clauses name a structure"]
    EXC --> LOGITS
    SAM --> CENTROID
    NULL --> VALID

    classDef input fill:#e8eaf6,stroke:#3949ab,stroke-width:2px,color:#1a1a1a;
    classDef optional fill:#fafafa,stroke:#9e9e9e,stroke-dasharray:4 3,color:#1a1a1a;
    classDef frozen fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px,color:#1a1a1a;
    classDef free fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1.5px,color:#1a1a1a;
    classDef trained fill:#fbe9e7,stroke:#e64a19,stroke-width:1.5px,color:#1a1a1a;
    classDef tensor fill:#fffde7,stroke:#f9a825,stroke-width:1px,color:#1a1a1a;
    classDef out fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1a1a1a;

    class IMG,NID,DID input;
    class CACHE,SWAP,ALPHA optional;
    class ENC,VAL,POS,KEY,NP,QRY,MHA,LN,DEC,MH,SIG frozen;
    class CEN,MAR,FLD,PRD,WMS,RES,EXC,SAM free;
    class D0,D1,D2,U0,U1,STEM,BLK,UP,SKIP,HEAD,HM,NULL trained;
    class ANCH,BI,CAT tensor;
    class LOGITS,CENTROID,VALID out;
```

**Legend.** Blue boxes are frozen: Stage A never trains here. Purple boxes are parameter-free and carry no gradient: the mapper, the rescaling, the exclusion, the soft-argmax. Orange boxes are the 268,275 trainable parameters: `B(I)`, the carver and the null head. Yellow boxes are tensors that cross a module boundary. Dashed boxes are optional inputs or ablations.

**What the picture enforces**
- `name_ids` has exactly one exit, into Stage A. Downstream of `A_i` no name exists.
- `direction_ids` has exactly one exit, into the mapper. The carver receives the direction only as the *shape* of `F_i`.
- `B(I)` has one input, the image, so it knows nothing of the prompt.
- The null head reads four numbers and no pixels.
- Stage A's feature pyramid (`ENC`, `DEC`) never leaves its box. Only the detached probabilities do.
