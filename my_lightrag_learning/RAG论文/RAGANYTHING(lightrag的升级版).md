# RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK

Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang∗

The University of Hong Kong

zrguo101@hku.hk xubinrencs@gmail.com chaohuang75@gmail.com

# ABSTRACT

Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigmfor expanding Large Language Models beyond their static training limitations.However, a critical misalignment exists between current RAG capabilities andreal-world information environments. Modern knowledge repositories are inher-ently multimodal, containing rich combinations of textual content, visual elements,structured tables, and mathematical expressions. Yet existing RAG frameworks arelimited to textual content, creating fundamental gaps when processing multimodaldocuments. We present RAG-Anything, a unified framework that enables compre-hensive knowledge retrieval across all modalities. Our approach reconceptualizesmultimodal content as interconnected knowledge entities rather than isolated datatypes. The framework introduces dual-graph construction to capture both cross-modal relationships and textual semantics within a unified representation. Wedevelop cross-modal hybrid retrieval that combines structural knowledge naviga-tion with semantic matching. This enables effective reasoning over heterogeneouscontent where relevant evidence spans multiple modalities. RAG-Anything demon-strates superior performance on challenging multimodal benchmarks, achievingsignificant improvements over state-of-the-art methods. Performance gains becomeparticularly pronounced on long documents where traditional approaches fail. Ourframework establishes a new paradigm for multimodal knowledge access, eliminat-ing the architectural fragmentation that constrains current systems. Our frameworkis open-sourced at: https://github.com/HKUDS/RAG-Anything.

# 1 INTRODUCTION

Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expandingthe knowledge boundaries of Large Language Models (LLM) beyond their static training limita-tions Zhang et al. (2025). By enabling dynamic retrieval and incorporation of external knowledgeduring inference, RAG systems transform static language models into adaptive, knowledge-awaresystems. This capability has proven essential for applications requiring up-to-date information,domain-specific knowledge, or factual grounding that extends beyond pre-training corpora.

However, existing RAG frameworks focus exclusively on text-only knowledge while neglecting therich multimodal information present in real-world documents. This limitation fundamentally mis-aligns with how information exists in authentic environments. Real-world knowledge repositories areinherently heterogeneous and multimodal Abootorabi et al. (2025). They contain rich combinationsof textual content, visual elements, structured tables, and mathematical expressions across diversedocument formats. This textual assumption forces existing RAG systems to either discard non-textualinformation entirely or flatten complex multimodal content into inadequate textual approximations.

The consequences of this limitation become particularly severe in document-intensive domainswhere multimodal content carries essential meaning. Academic research, financial analysis, andtechnical documentation represent prime examples of knowledge-rich environments. These domainsfundamentally depend on visual and structured information. Critical insights are often encodedexclusively in non-textual formats. Such formats resist meaningful conversion to plain text.

The consequences of this limitation become particularly severe in knowledge-intensive domains wheremultimodal content carries essential meaning. Three representative scenarios illustrate the critical

need for multimodal RAG capabilities. In Scientific Research, experimental results are primarilycommunicated through plots, diagrams, and statistical visualizations. These contain core discoveriesthat remain invisible to text-only systems. Financial Analysis relies heavily on market charts,correlation matrices, and performance tables. Investment insights are encoded in visual patternsrather than textual descriptions. Additionally, Medical Literature Analysis depends on radiologicalimages, diagnostic charts, and clinical data tables. These contain life-critical information essential foraccurate diagnosis and treatment decisions. Current RAG frameworks systematically exclude thesevital knowledge sources across all three scenarios. This creates fundamental gaps that render theminadequate for real-world applications requiring comprehensive information understanding. Therefore,multimodal RAG emerges as a critical advancement. It is necessary to bridge these knowledge gapsand enable truly comprehensive intelligence across all modalities of human knowledge representation.

Addressing multimodal RAG presents three fundamental technical challenges that demand principledsolutions. This makes it significantly more complex than traditional text-only approaches. The naivesolution of converting all multimodal content to textual descriptions introduces severe informationloss. Visual elements such as charts, diagrams, and spatial layouts contain semantic richness thatcannot be adequately captured through text alone. These inherent limitations necessitate the design ofeffective technical components. Such components must be specifically designed to handle multimodalcomplexity and preserve the full spectrum of information contained within diverse content types.

Technical Challenges. • First, the unified multimodal representation challenge requires seam-lessly integrating diverse information types. The system must preserve their unique characteristicsand cross-modal relationships. This demands advanced multimodal encoders that can capture bothintra-modal and inter-modal dependencies without losing essential visual semantics. • Second, thestructure-aware decomposition challenge demands intelligent parsing of complex layouts. Thesystem must maintain spatial and hierarchical relationships crucial for understanding. This requiresspecialized layout-aware parsing modules that can interpret document structure and preserve contex-tual positioning of multimodal elements. • Third, the cross-modal retrieval challenge necessitatessophisticated mechanisms that can navigate between different modalities. These mechanisms mustreason over their interconnections during retrieval. This calls for cross-modal alignment systemscapable of understanding semantic correspondences across text, images, and structured data. Thesechallenges are amplified in long-context scenarios. Relevant evidence is dispersed across multiplemodalities and sections, requiring coordinated reasoning across heterogeneous information sources.

Our Contributions. To address these challenges, we introduce RAG-Anything, a unified frameworkthat fundamentally reimagines multimodal knowledge representation and retrieval. Our approachemploys a dual-graph construction strategy that elegantly bridges the gap between cross-modalunderstanding and fine-grained textual semantics. Rather than forcing diverse modalities into text-centric pipelines, RAG-Anything constructs complementary knowledge graphs that preserve bothmultimodal contextual relationships and detailed textual knowledge. This design enables seamlessintegration of visual elements, structured data, and mathematical expressions within a unified retrievalframework. The system maintains semantic integrity across modalities while ensuring efficientcross-modal reasoning capabilities throughout the process.

Our cross-modal hybrid retrieval mechanism strategically combines structural knowledge nav-igation with semantic similarity matching. This architecture addresses the fundamental limita-tion of existing approaches that rely solely on embedding-based retrieval or keyword matching.RAG-Anything leverages explicit graph relationships to capture multi-hop reasoning patterns. Itsimultaneously employs dense vector representations to identify semantically relevant content thatlacks direct structural connections. The framework introduces modality-aware query processingand cross-modal alignment systems. These enable textual queries to effectively access visual andstructured information. This unified approach eliminates the architectural fragmentation that plaguescurrent multimodal RAG systems. It delivers superior performance particularly on long-contextdocuments where relevant evidence spans multiple modalities and document sections.

Experimental Validation. To validate the effectiveness of our proposed approach, we conduct com-prehensive experiments on two challenging multimodal benchmarks: DocBench and MMLongBench.Our evaluation demonstrates that RAG-Anything achieves superior performance across diverse do-mains. The framework represents substantial improvements over state-of-the-art baselines. Notably,our performance gains become increasingly significant as content length increases. We observeparticularly pronounced advantages on long-context materials. This validates our core hypothesis

that dual-graph construction and cross-modal hybrid retrieval are essential for handling complexmultimodal materials. Our ablation studies reveal that graph-based knowledge representation providesthe primary performance gains. Traditional chunk-based approaches fail to capture the structuralrelationships critical for multimodal reasoning. Case studies further demonstrate that our frameworkexcels at precise localization within complex layouts. The system effectively disambiguates similarterminology and navigates multi-panel visualizations through structure-aware retrieval mechanisms.

# 2 THE RAG-ANYTHING FRAMEWORK

# 2.1 PRELIMINARY

Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for dynamicallyexpanding the knowledge boundaries of LLMs. While LLMs demonstrate exceptional reasoningcapabilities, their knowledge remains static and bounded by training data cutoffs. This creates anever-widening gap with the rapidly evolving information landscape. RAG systems address this criticallimitation by enabling LLMs to retrieve and incorporate external knowledge sources during inference.This transforms them from static repositories into adaptive, knowledge-aware systems.

The Multimodal Reality: Beyond Text-Only RAG. Current RAG systems face a critical limitationthat severely restricts their real-world deployment. Existing frameworks operate under the restrictiveassumption that knowledge corpus consists exclusively of plain textual documents. This assump-tion fundamentally misaligns with how information exists in authentic environments. Real-worldknowledge repositories are inherently heterogeneous and multimodal, containing rich combinationsof textual content, visual elements, structured data, and mathematical expressions. These diverseknowledge sources span multiple document formats and presentation mediums, from research papersand technical slides to web pages and interactive documents.

# 2.1.1 MOTIVATING RAG-ANYTHING

This multimodal reality introduces fundamental technical challenges that expose the inadequacy ofcurrent text-only RAG approaches. Effective multimodal RAG requires unified indexing strategiesthat can handle disparate data types, cross-modal retrieval mechanisms that preserve semanticrelationships across modalities, and sophisticated synthesis techniques that can coherently integratediverse information sources. These challenges demand a fundamentally different architecturalapproach rather than incremental improvements to existing systems.

The RAG-Anything framework introduces a unified approach for retrieving and processing knowl-edge from heterogeneous multimodal information sources. Our system addresses the fundamentalchallenge of handling diverse data modalities and document formats within a retrieval pipeline.The framework comprises three core components: universal indexing for multimodal knowledge,cross-modal adaptive retrieval, and knowledge-enhanced response generation. This integrated designenables effective knowledge utilization across modalities while maintaining computational efficiency.

# 2.2 UNIVERSAL REPRESENTATION FOR HETEROGENEOUS KNOWLEDGE

A key requirement for universal knowledge access is the ability to represent heterogeneous multimodalcontent in a unified, retrieval-oriented abstraction. Unlike existing pipelines that simply parsedocuments into text segments, RAG-Anything introduces Multimodal Knowledge Unification. Thisprocess decomposes raw inputs into atomic knowledge units while preserving their structural contextand semantic alignment. For instance, RAG-Anything ensures that figures remain grounded in theircaptions, equations remain linked to surrounding definitions, and tables stay connected to explanatorynarratives. This transforms heterogeneous files into a coherent substrate for cross-modal retrieval.

Formally, each knowledge source $k _ { i } \in \mathcal { K }$ (e.g., a web page) is decomposed into atomic content units:

$$
k _ {i} \xrightarrow {\text {D e c o m p o s e}} \left\{c _ {j} = \left(t _ {j}, x _ {j}\right) \right\} _ {j = 1} ^ {n _ {i}}, \tag {1}
$$

where each unit $c _ { j }$ consists of a modality type $t _ { j } \in$ text, image, table, equation, . . . and its corre-sponding raw content $x _ { j }$ . The content $x _ { j }$ represents the extracted information from the originalknowledge source, processed in a modality-aware manner to preserve semantic integrity.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/79d305ec95ae13155ae24774f90bcf3e48505e8195dad639318bda6230d9815a.jpg)



Figure 1: Overview of our proposed universal RAG framework RAG-Anything.


To ensure high-fidelity extraction, RAG-Anything leverages specialized parsers for different contenttypes. Text is segmented into coherent paragraphs or list items. Figures are extracted with associatedmetadata such as captions and cross-references. Tables are parsed into structured cells with headersand values. Mathematical expressions are converted into symbolic representations. The resulting $x _ { j }$preserves both content and structural context within the source. This provides a faithful, modality-consistent representation. The decomposition abstracts diverse file formats into atomic units whilemaintaining their hierarchical order and contextual relationships. This canonicalization enablesuniform processing, indexing, and retrieval of multimodal content within our framework.

# 2.2.1 DUAL-GRAPH CONSTRUCTION FOR MULTIMODAL KNOWLEDGE

While multimodal knowledge unification provides a uniform abstraction across modalities, directlyconstructing a single unified graph often risks overlooking modality-specific structural signals. Theproposed RAG-Anything addresses this challenge through a dual-graph construction strategy. Thesystem first builds a cross-modal knowledge graph that faithfully grounds non-textual modalitieswithin their contextual environment. It then constructs a text-based knowledge graph using es-tablished text-centric extraction pipelines. These complementary graphs are merged through entityalignment. This design ensures accurate cross-modal grounding and comprehensive coverage oftextual semantics, enabling richer knowledge representation and robust retrieval.

• Cross-Modal Knowledge Graph: Non-textual content like images, tables, and equations containsrich semantic information that traditional text-only approaches often overlook. To preserve thisknowledge, RAG-Anything constructs a multimodal knowledge graph where non-text atomicunits are transformed into structured graph entities. RAG-Anything leverages multimodal largelanguage models to derive two complementary textual representations from each atomic contentunit. The first is a detailed description $d _ { j } ^ { \mathrm { c h u n k } }$ optimized for cross-modal retrieval. The second isan entity summary eentityj $e _ { j } ^ { \mathrm { e n t i t y } }$ containing key attributes such as entity name, type, and description forgraph construction. The generation process is context-aware, processing each unit with its localneighborhood $C _ { j } = \{ c _ { k } \ | \ | k - j | \leq \delta \}$ , where $\delta$ controls the contextual window size. This ensuresrepresentations accurately reflect each unit’s role within the broader document structure.

Building on these textual representations, RAG-Anything constructs the graph structure using non-text units as anchor points. For each non-text unit $c _ { j }$ , the graph extraction routine $R ( \cdot )$ processesits description $d _ { j } ^ { \mathrm { c h u n k } }$ to identify fine-grained entities and relations:

$$
\left(\mathcal {V} _ {j}, \mathcal {E} _ {j}\right) = R \left(d _ {j} ^ {\text {c h u n k}}\right), \tag {2}
$$

where $\nu _ { j }$ and $\mathcal { E } _ { j }$ denote the sets of intra-chunk entities and their relations, respectively. Eachatomic non-text unit is associated with a multimodal entity node $v _ { j } ^ { \mathrm { m m } }$ that serves as an anchor for

its intra-chunk entities through explicit belongs_to edges:

$$
\tilde {V} = \left\{v _ {j} ^ {\mathrm {m m}} \right\} _ {j} \cup \bigcup_ {j} \mathcal {V} _ {j}, \tag {3}
$$

$$
\tilde {E} = \bigcup_ {j} \mathcal {E} _ {j} \cup \bigcup_ {j} \left\{\left(u \xrightarrow {\text {b e l o n g s} _ {\text {t o}}} v _ {j} ^ {\mathrm {m m}}\right): u \in \mathcal {V} _ {j} \right\}. \tag {4}
$$

This construction preserves modality-specific grounding while ensuring non-textual content is con-textualized by its textual neighborhood. This enables reliable cross-modal retrieval and reasoning.

• Text-based Knowledge Graph: For text modality chunks, we construct a traditional text-basedknowledge graph following established methodologies similar to LightRAG (Guo et al., 2024)and GraphRAG (Edge et al., 2024). The extraction process operates directly on textual content $x _ { j }$where $t _ { j } =$ text, leveraging named entity recognition and relation extraction techniques to identifyentities and their semantic relationships. Given the rich semantic information inherent in textualcontent, multimodal context integration is not required for this component. The resulting text-basedknowledge graph captures explicit knowledge and semantic connections present in textual portionsof documents, complementing the multimodal graph’s cross-modal grounding capabilities.

# 2.2.2 GRAPH FUSION AND INDEX CREATION

The separate cross-modal and text-based knowledge graphs capture complementary aspects ofdocument semantics. Integrating them creates a unified representation leveraging visual-textualassociations and fine-grained textual relationships for enhanced retrieval.

• (i) Entity Alignment and Graph Fusion. To create a unified knowledge representation, wemerge the multimodal knowledge graph $( \tilde { V } , \tilde { E } )$ and text-based knowledge graph through entity align-ment. This process uses entity names as primary matching keys to identify semantically equivalententities across both graph structures. The integration consolidates their representations, creatinga comprehensive knowledge graph $\mathcal { G } = ( \nu , \mathcal { E } )$ . This graph captures both multimodal contextualrelationships and text-based semantic connections. The merged graph provides a holistic view of thedocument collection. This enables effective retrieval by leveraging visual-textual associations fromthe multimodal graph and fine-grained textual knowledge relationships from the text-based graph.

• (ii) Dense Representation Generation. To enable efficient similarity-based retrieval, we constructa comprehensive embedding table $\tau$ that encompasses all components generated during the indexingprocess. We encode dense representations for all graph entities, relationships, and atomic contentchunks across modalities using an appropriate encoder. This creates a unified embedding space whereeach component $s \in$ entities, relations, chunks is mapped to its corresponding dense representation:

$$
\mathcal {T} = \operatorname {e m b} (s): s \in \mathcal {V} \cup \mathcal {E} \cup c _ {j j}, \tag {5}
$$

where $\mathrm { e m b } ( \cdot )$ denotes the embedding function tailored for each component type. Together, theunified knowledge graph $\mathcal { G }$ and the embedding table $\tau$ constitute the complete retrieval index${ \mathcal { T } } = ( { \mathcal { G } } , { \mathcal { T } } )$ . This provides both structural knowledge representation and dense vector space forefficient cross-modal similarity search during the subsequent retrieval stage.

# 2.3 CROSS-MODAL HYBRID RETRIEVAL

The retrieval stage operates on the index $\mathcal { I } = ( \mathcal { G } , \mathcal { T } )$ to identify relevant knowledge components for agiven user query. Traditional RAG methods face significant limitations when dealing with multimodaldocuments. They typically rely on semantic similarity within single modalities and fail to capture therich interconnections between visual, mathematical, tabular, and textual elements. To address thesechallenges, our framework introduces a cross-modal hybrid retrieval mechanism. This mechanismleverages structural knowledge and semantic representations across heterogeneous modalities.

Modality-Aware Query Encoding. Given a user query $q$ , we first perform modality-aware queryanalysis to extract lexical cues and potential modality preferences embedded within the query.For instance, queries containing terms such as "figure," "chart," "table," or "equation" provideexplicit signals about the expected modality of relevant information. We then compute a unified textembedding ${ \mathbf { e } _ { q } }$ using the same encoder employed during indexing, ensuring consistency between

query and knowledge representations. This embedding-based approach enables cross-modal retrievalcapabilities where textual queries can effectively access multimodal content through their sharedrepresentations, maintaining retrieval consistency while preserving cross-modal accessibility.

Hybrid Knowledge Retrieval Architecture. Recognizing that knowledge relevance manifeststhrough both explicit structural connections and implicit semantic relationships, we design a hybridretrieval architecture that strategically combines two complementary mechanisms.

• (i) Structural Knowledge Navigation. This mechanism addresses the challenge of capturingexplicit relationships and multi-hop reasoning patterns. Traditional keyword-based retrieval oftenfails to identify knowledge connected through intermediate entities or cross-modal relationships. Toovercome this limitation, we exploit the structural properties encoded within our unified knowledgegraph G. We employ keyword matching and entity recognition to locate relevant graph components.The retrieval process begins with exact entity matching against query terms.

We then perform strategic neighborhood expansion to include related entities and relationships withina specified hop distance. This structural approach proves particularly effective at uncovering high-level semantic connections and entity-relation patterns that span multiple modalities. It capitalizeson the rich cross-modal linkages established in our multimodal knowledge graph. The structuralnavigation yields candidate set $\mathcal { C } _ { \mathrm { s t r u } } ( { q } )$ containing relevant entities, relationships, and their associatedcontent chunks that provide comprehensive contextual information.

• (ii) Semantic Similarity Matching. This mechanism addresses the challenge of identifyingsemantically relevant knowledge that lacks explicit structural connections. While structural navigationexcels at following explicit relationships, it may miss relevant content that is semantically related butnot directly connected in the graph topology. To bridge this gap, we conduct dense vector similaritysearch between the query embedding $\mathbf { e } _ { q }$ and all components stored in embedding table $\tau$ .

This approach encompasses atomic content chunks across all modalities, graph entities, and relation-ship representations, enabling fine-grained semantic matching that can surface relevant knowledgeeven when traditional lexical or structural signals are absent. The learned embedding space capturesnuanced semantic relationships and contextual similarities that complement the explicit structuralsignals from the navigation mechanism. This retrieval pathway returns the top-k most semanticallysimilar chunks $\mathcal { C } _ { \mathrm { s e m a n } } ( q )$ ranked by cosine similarity scores, ensuring comprehensive coverage ofboth structurally and semantically relevant knowledge.

Candidate Pool Unification. Both retrieval pathways may return overlapping candidates withdiffering relevance signals. This necessitates a principled approach to unify and rank results. Retrievalcandidates from both pathways are unified into a comprehensive candidate pool: $\mathcal { C } ( q ) = \mathcal { C } _ { \mathrm { s t r u } } ( q ) \cup$$\mathcal { C } _ { \mathrm { s e m a n } } ( q )$ . Simply merging candidates would ignore distinct evidence each pathway provides. Itwould fail to account for redundancy between retrieved content.

• (i) Multi-Signal Fusion Scoring. To address these challenges, we apply a sophisticated fusionscoring mechanism integrating multiple complementary relevance signals. These include structuralimportance derived from graph topology, semantic similarity scores from embedding space, and query-inferred modality preferences obtained through lexical analysis. This multi-faceted scoring approachensures that final ranked candidates $\mathcal { C } ^ { \star } ( q )$ effectively balance structural knowledge relationships withsemantic relevance while appropriately weighting different modalities based on query characteristics.

• (ii) Hybrid Retrieval Integration. The resulting hybrid retrieval mechanism enables our frameworkto leverage the complementary strengths of both knowledge graphs and dense representations. Thisprovides comprehensive coverage of relevant multimodal knowledge for response generation.

# 2.4 FROM RETRIEVAL TO SYNTHESIS

Effective multimodal question answering requires preserving rich visual semantics while maintainingcoherent grounding across heterogeneous knowledge sources. Simple text-only approaches losecrucial visual information, while naive multimodal methods struggle with coherent cross-modalintegration. Our synthesis stage addresses these challenges by systematically combining retrievedmultimodal knowledge into comprehensive, evidence-grounded responses.

• (i) Building Textual Context. Given the top-ranked retrieval candidates $\mathcal { C } ^ { \star } ( q )$ , we construct astructured textual context. We concatenate textual representations of all retrieved components, includ-


Table 1: Statistics of Experimental Datasets.


<table><tr><td>Dataset</td><td># Documents</td><td># Avg. Pages</td><td># Avg. Tokens</td><td># Doc Types</td><td># Questions</td></tr><tr><td>DocBench</td><td>229</td><td>66</td><td>46377</td><td>5</td><td>1102</td></tr><tr><td>MMLongBench</td><td>135</td><td>47.5</td><td>21214</td><td>7</td><td>1082</td></tr></table>

ing entity summaries, relationship descriptions, and chunk contents. The concatenation incorporatesappropriate delimiters to indicate modality types and hierarchical origins. This approach ensures thelanguage model can effectively parse and reason over heterogeneous knowledge components.

• (ii) Recovering Visual Content. For multimodal chunks corresponding to visual artifacts, weperform dereferencing to recover original visual content, creating $\mathcal { V } ^ { \star } ( q )$ . This design maintains con-sistency with our unified embedding strategy. Textual proxies enable efficient retrieval while authenticvisual content provides rich semantics necessary for sophisticated reasoning during synthesis.

The synthesis process jointly conditions on both the assembled comprehensive textual context anddereferenced visual artifacts using a vision-language model:

$$
\operatorname {R e s p o n s e} = \operatorname {V L M} (q, \mathcal {P} (q), \mathcal {V} ^ {\star} (q)), \tag {6}
$$

where the VLM integrates information from query, textual context, and visual content. This unifiedconditioning enables sophisticated visual interpretation while maintaining grounding in retrievedevidence. The resulting responses are both visually informed and factually grounded.

# 3 EVALUATION

# 3.1 EXPERIMENTAL SETTINGS

Evaluation Datasets. We conduct comprehensive evaluations on two challenging multimodalDocument Question Answering (DQA) benchmarks that reflect real-world complexity and diversity.DocBench (Zou et al., 2024) provides a rigorous testbed with 229 multimodal documents spanningfive critical domains: Academia, Finance, Government, Laws, and News. The dataset includes 1,102expert-crafted question-answer pairs. These documents are notably extensive, averaging 66 pages andapproximately 46,377 tokens, which presents substantial challenges for long-context understanding.

MMLongBench (Ma et al., 2024) complements this evaluation by focusing specifically on long-context multimodal document comprehension. It features 135 documents across 7 diverse documenttypes with 1,082 expert-annotated questions. Together, these benchmarks provide comprehensivecoverage of the multimodal document understanding challenges that RAG-Anything aims to address.They ensure our evaluation captures both breadth across domains and depth in document complexity.Detailed dataset statistics and characteristics are provided in Appendix A.1.

Baselines. We compare RAG-Anything against the following methods for performance evaluation:

• GPT-4o-mini: A powerful multimodal language model with native text and image understandingcapabilities. Its 128K token context window enables direct processing of entire documents. Weevaluate this model as a strong baseline for long-context multimodal understanding.

• LightRAG (Guo et al., 2024): A graph-enhanced RAG system that integrates structured knowledgerepresentation with dual-level retrieval mechanisms. It captures both fine-grained entity-relationinformation and broader semantic context, improving retrieval precision and response coherence.

• MMGraphRAG (Wan & Yu, 2025): A multimodal retrieval framework that constructs unifiedknowledge graphs spanning textual and visual content. This method employs spectral clusteringfor multimodal entity analysis and retrieves context along reasoning paths to guide generation.

Experimental Settings. In our experiments, we implement all baselines using GPT-4o-mini asthe backbone LLM. Documents are parsed using MinerU (Wang et al., 2024) to extract text, im-ages, tables, and equations for downstream RAG processing. For the retrieval pipeline, we em-ploy the text-embedding-3-large model with 3072-dimensional embeddings. We use thebge-reranker-v2-m3 model for reranking. For graph-based RAG methods, we enforce a com-bined entity-and-relation token limit of 20,000 tokens and a chunk token limit of 12,000 tokens.


Table 2: Accuracy $( \% )$ on DocBench Dataset. Performance results with best scores highlighted indark blue and second-best in light blue. Domain categories include Academia (Aca.), Finance(Fin.), Government (Gov.), Legal Documents (Law), and News Articles (News). Document types arecategorized as Text-only (Txt.), Multimodal (Mm.), and Unanswerable queries (Una.).


<table><tr><td rowspan="2">Method</td><td colspan="5">Domains</td><td colspan="3">Types</td><td rowspan="2">Overall</td></tr><tr><td>Aca.</td><td>Fin.</td><td>Gov.</td><td>Law.</td><td>News</td><td>Txt.</td><td>Mm.</td><td>Una.</td></tr><tr><td>GPT-4o-mini</td><td>40.3</td><td>46.9</td><td>60.3</td><td>59.2</td><td>61.0</td><td>61.0</td><td>43.8</td><td>49.6</td><td>51.2</td></tr><tr><td>LightRAG</td><td>53.8</td><td>56.2</td><td>59.5</td><td>61.8</td><td>65.7</td><td>85.0</td><td>59.7</td><td>46.8</td><td>58.4</td></tr><tr><td>MMGraphRAG</td><td>64.3</td><td>52.8</td><td>64.9</td><td>40.0</td><td>61.5</td><td>67.6</td><td>66.0</td><td>60.5</td><td>61.0</td></tr><tr><td>RAGAnything</td><td>61.4</td><td>67.0</td><td>61.5</td><td>60.2</td><td>66.3</td><td>85.0</td><td>76.3</td><td>46.0</td><td>63.4</td></tr></table>


Table 3: Accuracy $( \% )$ on MMLongBench across different domains and overall performance. Best re-sults are highlighted in dark blue and second-best in light blue.. Domain categories include ResearchReports/Introductions (Res.), Tutorials/Workshops (Tut.), Academic Papers (Acad.), Guidebooks(Guid.), Brochures (Broch.), Administration/Industry Files (Admin.), and Financial Reports (Fin.).


<table><tr><td rowspan="2">Method</td><td colspan="7">Domains</td><td rowspan="2">Overall</td></tr><tr><td>Res.</td><td>Tut.</td><td>Acad.</td><td>Guid.</td><td>Broch.</td><td>Admin.</td><td>Fin.</td></tr><tr><td>GPT-4o-mini</td><td>35.5</td><td>44.0</td><td>24.6</td><td>33.1</td><td>29.5</td><td>46.8</td><td>31.1</td><td>33.5</td></tr><tr><td>LightRAG</td><td>40.8</td><td>34.1</td><td>36.2</td><td>39.4</td><td>41.0</td><td>44.4</td><td>38.3</td><td>38.9</td></tr><tr><td>MMGraphRAG</td><td>40.8</td><td>36.5</td><td>35.7</td><td>35.8</td><td>28.2</td><td>46.9</td><td>38.5</td><td>37.7</td></tr><tr><td>RAGAnything</td><td>46.6</td><td>43.5</td><td>38.7</td><td>43.9</td><td>34.0</td><td>45.7</td><td>43.6</td><td>42.8</td></tr></table>

Outputs are constrained to a one-sentence format. For the baseline GPT-4o-mini in our QA scenario,documents are concatenated into image form with a maximum of 50 pages per document, rendered at144 dpi. Finally, all query results are evaluated for accuracy by GPT-4o-mini.

# 3.2 PERFORMANCE COMPARISON

Superior Performance and Cross-Domain Generalization. RAG-Anything demonstrates superioroverall performance over baselines through its unified multimodal framework. Unlike LightRAG,which is restricted to text-only content processing, RAG-Anything treats text, images, tables, andequations as first-class entities. MMGraphRAG only adds basic image processing while treatingtables and equations as plain text, missing crucial structural information. RAG-Anything introducesa comprehensive dual-graph construction strategy that preserves structural relationships across allmodalities. This unified approach enables superior performance across both evaluation benchmarks.

Enhanced Long-Context Performance. RAG-Anything demonstrates superior performance onlong-context documents. The framework excels where relevant evidence is dispersed across multiplemodalities and sections. It achieves the best results in information-dense domains such as ResearchReports and Financial Reports on MMLongBench. These improvements stem from the structuredcontext injection mechanism. This mechanism integrates dual-graph construction for cross-page entityalignment. It combines semantic retrieval with structural navigation. The framework also employsmodality-aware processing for efficient context window utilization. Unlike baselines that cannotuniformly process diverse modalities, RAG-Anything effectively captures scattered multimodalevidence. Its cross-modal hybrid retrieval architecture combines structural knowledge navigationwith semantic similarity matching. This enables the framework to leverage both explicit relationshipsand implicit semantic connections across modalities.

To systematically evaluate model performance across varying document lengths, we conductedcomprehensive experiments on both datasets. As illustrated in Figure 2, RAG-Anything and MM-GraphRAG exhibit comparable performance on shorter documents. However, RAG-Anything’sadvantages become increasingly pronounced as document length grows. On DocBench, the perfor-mance gap expands dramatically to over 13 points for documents exceeding 100 pages $6 8 . 2 \%$ vs.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/ee5785bf216b1f1ff615384c72547c777bf46804bbf3fe634e3965a1465bc444.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/0ff7dfded638c32c52f0c49484c85a6a53ac5bf3523d2e039357c7c3273a021a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/8b03d3d346839fe45a7bbe8bf5c15bee1bfd8c4582617bfcd4d5bd9e11de5d2b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/90e61090bca3d83abffc0c5f10a259fdea55fe1e39646bf9336eb7761bc63982.jpg)



Figure 2: Performance evaluation across documents of varying lengths.



Table 4: Ablation study results on DocBench. The “Chunk-only” variant bypasses dual-graphconstruction and relies solely on traditional chunk-based retrieval, while “w/o Reranker” eliminatescross-modal reranking but preserves the core graph-based architecture.


<table><tr><td rowspan="2">Method</td><td colspan="5">Domains</td><td colspan="3">Types</td><td rowspan="2">Overall</td></tr><tr><td>Aca.</td><td>Fin.</td><td>Gov.</td><td>Law.</td><td>News</td><td>Txt.</td><td>Mm.</td><td>Una.</td></tr><tr><td>Chunk-only</td><td>55.8</td><td>61.5</td><td>60.1</td><td>60.7</td><td>64.0</td><td>81.6</td><td>66.2</td><td>43.5</td><td>60.0</td></tr><tr><td>w/o Reranker</td><td>60.9</td><td>63.5</td><td>58.8</td><td>60.2</td><td>68.6</td><td>81.7</td><td>74.7</td><td>45.4</td><td>62.4</td></tr><tr><td>RAGAnything</td><td>61.4</td><td>67.0</td><td>61.5</td><td>60.2</td><td>66.3</td><td>85.0</td><td>76.3</td><td>46.0</td><td>63.4</td></tr></table>

$5 4 . 6 \%$ for 101–200 pages; $6 8 . 8 \%$ vs. $5 5 . 0 \%$ for $^ { 2 0 0 + }$ pages). On MMLongBench, RAG-Anythingdemonstrates consistent improvements across all length categories, achieving accuracy gains of 3.4points for 11–50 pages, 9.3 points for 51–100 pages, and 7.9 points for 101–200 pages. Thesefindings confirm that our dual-graph construction and cross-modal hybrid retrieval mechanism isparticularly effective for long-document reasoning tasks.

# 3.3 ARCHITECTURAL VALIDATION WITH ABLATION STUDIES

To isolate and quantify the contributions of key architectural components in RAG-Anything, weconducted systematic ablation studies examining two critical design choices. Given that our approachfundamentally differs from existing methods through dual-graph construction and hybrid retrieval,we specifically evaluated: i) Chunk-only, which bypasses graph construction entirely and reliessolely on traditional chunk-based retrieval, and ii) w/o Reranker, which eliminates the cross-modalreranking component while preserving the core graph-based architecture.

As demonstrated in Table 4, the results validate our architectural design through striking performancevariations. • Graph Construction is Essential. The chunk-only variant achieves merely $6 0 . 0 \%$accuracy with substantial cross-domain drops. This demonstrates that traditional chunking fails tocapture structural and cross-modal relationships essential for multimodal documents. • RerankingProvides Marginal Gains. Removing the reranker yields only a modest decline to $6 2 . 4 \%$ , while thefull model achieves $6 3 . 4 \%$ accuracy. This indicates that cross-modal reranking provides valuablerefinement, but primary gains stem from our graph-based retrieval and cross-modal integration.

# 3.4 CASE STUDIES

Multimodal documents contain rich structural information within each modality. Understandingthese intra-modal structures is crucial for accurate reasoning. We analyze two representative casesfrom DocBench to demonstrate how RAG-Anything leverages these structures. These cases highlighta key limitation of existing methods. Baselines either rely on superficial textual cues or flattencomplex visual elements into plain text. In contrast, RAG-Anything builds modality-aware graphsthat preserve essential relationships (e.g., table header cell unit edges; pane caption axisedges). This enables precise reasoning over complex document layouts.

• Case 1: Multi-panel Figure Interpretation. This case examines a common scenario in academicliterature. Researchers often need to compare results across different experimental conditions. Theseresults are typically presented in multi-panel visualizations. Figure 3 shows a challenging t-SNE

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/5abdb40d6c5176c729250cae9c21a6ca90e330a217e51b6f7238aeca93f2de5d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/5bb382d86ae1c16bea75346062629dab147569e9baf0d5b6f3c8dd01e1f738e8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/fdd25aeb3faaa5a667553e3621c1404b3d0be7811b24638b1382584091ad29d6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/bcb4ce14f67622e6f93a0b52cb318b9b8d7819fa1765123d207596c85a9f081b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/339f8a81f5056da5479ef86f30bc4548b82b0cecc47a1310462d0a0d8a9b326c.jpg)



Figure 3: Multi-panel figure interpretation case. The query requires identifying cluster separationpatterns from the style-space panel, while avoiding confusion from the adjacent content-space panel.


visualization with multiple subpanels. The query requires distinguishing between two related butdistinct panels. RAG-Anything constructs a visual-layout graph where panels, axis titles, legends,and captions become nodes. Key edges encode semantic relationships. Panels contain specific plots.Captions provide contextual information. Subfigures relate hierarchically. This structure guides theretriever to focus on the style-space panel for comparing cluster separation patterns. The systemavoids confusion from the adjacent content space panel. This panel shows less clear distinctions.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/cbd2550a02567feeb0154742abbecc478f248183aab38eba49fb372dc65be619.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/1fdfc08f3890bb48caea6922f54939c67e1af1fac6617a11a0f2996453d9fa9b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/01e1478504685d8dbcae275c2716f848420e9e291c6986d589fc4678edccbcca.jpg)



MMGraphRAG??:Novo Nordisk spent a total of11,503 million DKK on wagesand salaries in 2020.



RAG-Anything(Correct??):Novo Nordisk's total amount spent on wagesand salaries in 2020 was DKK 26,778 million.



Figure 4: Financial table navigation case. The query involves locating the specific intersection of“Wages and salaries” row and $" 2 0 2 0 "$ column amid similar terminological entries.


• Case 2: Financial Table Navigation. This case addresses a common challenge in financialdocument analysis. Analysts must extract specific metrics from tables with similar terminologyand multiple time periods. Figure 4 shows this scenario. The query involves resolving ambiguousfinancial terms and selecting the correct column for a specified year.

RAG-Anything transforms the financial report table into a structured graph. Each row header, columnheader (year), data cell, and unit becomes a node. The edges capture key relationships: row-of,column-of, header-applies-to, and unit-of. This structure enables precise navigation. The retrieverfocuses on the row “Wages and salaries” and the column for $" 2 0 2 0 "$ . It directs attention to thetarget cell (26,778 million). The system successfully disambiguates nearby entries like “Share-basedpayments.” Competing methods treat tables as linear text. They often confuse numerical spans andyears. This leads to significantly inaccurate answers. RAG-Anything explicitly models relationshipswithin the table. It achieves precise selection and numeric grounding. This ensures accurate responses.

• Key Insights. Both cases demonstrate how RAG-Anything’s structure-aware design deliverstargeted advantages. Our approach transforms documents into explicit graph representations. Thesegraphs capture intra-modal relationships that traditional methods miss. In figures, connectionsbetween panels, captions, and axes enable panel-level comparisons. This goes beyond keywordmatching. In tables, row–column–unit graphs ensure accurate identification through modeling.

This structure-aware retrieval design reduces confusion from repeated terminology and complexlayouts. Traditional RAG systems struggle with these scenarios due to lack of structural understanding.Even MMGraphRAG fails here because it only considers image modality entities. It ignores othermodality entities like table cells, row headers, and column headers. RAG-Anything’s comprehensivegraph representation captures all modality-specific entities and their relationships. This enablesprecise, modality-specific grounding that leads to consistent improvements in document Q&A tasksrequiring fine-grained localization. Additional cases are available in Appendix A.2.

# 4 RELATED WORK

• Graph-Enhanced Retrieval-Augmented Generation. Large language models struggle withlong-context inputs and multi-hop queries, failing to precisely locate dispersed evidence (Zhang et al.,

2025). Graph structures address this limitation by introducing explicit relational modeling, improvingboth retrieval efficiency and reasoning accuracy (Bei et al., 2025).

Since GraphRAG (Edge et al., 2024), research has evolved along two complementary directions.First, graph construction approaches optimize structures for retrieval efficiency, ranging from Ligh-tRAG’s (Guo et al., 2024) sparsified indices to neural models like GNN-RAG (Mavromatis & Karypis,2024) and memory-augmented variants like HippoRAG (Jimenez Gutierrez et al., 2024). Second,knowledge aggregation approaches integrate information for multi-level reasoning through hier-archical methods like RAPTOR (Sarthi et al., 2024) and ArchRAG (Wang et al., 2025). Despitethese advances, existing systems remain text-centric with homogeneous inputs. This limits theirapplicability to multimodal documents and constrains robust reasoning over heterogeneous content.RAG-Anything addresses this gap by extending GraphRAG to all modalities.

• Multimodal Retrieval-Augmented Generation. Multimodal RAG represents a natural evolutionfrom text-based RAG systems, addressing the need to integrate external knowledge from diversedata modalities for comprehensive response generation (Abootorabi et al., 2025). However, currentapproaches are fundamentally constrained by their reliance on modality-specific architectures. Exist-ing methods demonstrate these constraints across domains: VideoRAG (Ren et al., 2025) employsdual-channel architectures for video understanding while MM-VID (Lin et al., 2023) converts videosto text, losing visual information; VisRAG (Yu et al., 2025) preserves document layouts as imagesbut misses granular relationships; MMGraphRAG (Wan & Yu, 2025) links scene graphs with textualrepresentations but suffers from structural blindness—treating tables and formulas as plain textwithout proper entity extraction, losing structural information for reasoning.

The fundamental problem underlying these limitations is architectural fragmentation. Current systemsrequire specialized processing pipelines for each modality. This creates poor generalizability as newmodalities demand custom architectures and fusion mechanisms. Such fragmentation introducescross-modal alignment difficulties, modality biases, and information bottlenecks. These issuessystematically compromise system performance and scalability. RAG-Anything addresses thisfragmentation through a unified graph-based framework. Our approach processes all modalities withconsistent structured modeling. This eliminates architectural constraints while preserving multimodalinformation integrity. The result is seamless cross-modal reasoning across heterogeneous content.

# 5 CONCLUSION

RAG-Anything introduces a paradigm shift in multimodal retrieval through its unified graph-basedframework. Our core technical innovation is the dual-graph construction strategy that seamlesslyintegrates cross-modal and text-based knowledge graphs. Rather than forcing diverse modalities intotext-centric pipelines that lose critical structural information, our approach fundamentally reconcep-tualizes multimodal content as interconnected knowledge entities with rich semantic relationships.The hybrid retrieval mechanism strategically combines structural navigation with semantic matching,enabling precise reasoning over complex document layouts. Comprehensive evaluation demonstratessuperior performance on long-context documents, particularly those exceeding 100 pages wheretraditional methods fail. This work establishes a new foundation for multimodal RAG systems thatcan handle the heterogeneous nature of diverse information landscapes.

Our analysis in Appendix A.5 reveals critical challenges facing current multimodal RAG systems.Two fundamental issues emerge through systematic failure case examination. First, systems exhibittext-centric retrieval bias, preferentially accessing textual sources even when queries explicitlyrequire visual information. Second, rigid spatial processing patterns fail to adapt to non-standarddocument layouts. These limitations manifest in cross-modal misalignment scenarios and structurallyambiguous tables. The findings highlight the need for adaptive spatial reasoning and layout-awareparsing mechanisms to handle real-world multimodal document complexity.

# REFERENCES



Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammad-khani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and EhsaneddinAsgari. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented genera-tion. arXiv preprint arXiv:2502.08826, 2025.





Yuanchen Bei, Weizhi Zhang, Siwen Wang, Weizhi Chen, Sheng Zhou, Hao Chen, Yong Li, Jiajun Bu,Shirui Pan, Yizhou Yu, et al. Graphs meet ai agents: Taxonomy, progress, and future opportunities.arXiv preprint arXiv:2506.18019, 2025.





Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: Agraph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.





Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation. arXiv preprint arXiv:2410.05779, 2024.





Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neuro-biologically inspired long-term memory for large language models. NeurIPS, 37:59532–59569,2024.





Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, JianfengWang, Lin Liang, Zicheng Liu, Yumao Lu, Ce Liu, and Lijuan Wang. Mm-vid: Advancing videounderstanding with gpt-4v(ision). arXiv preprint arXiv:2310.19773, 2023.





Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, YanMa, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understandingwith visualizations. Advances in Neural Information Processing Systems, 37:95963–96010, 2024.





Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language modelreasoning. arXiv preprint arXiv:2405.20139, 2024.





Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Vide-orag: Retrieval-augmented generation with extreme long-context videos. arXiv preprintarXiv:2502.01549, 2025.





Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning.Raptor: Recursive abstractive processing for tree-organized retrieval. In The Twelfth InternationalConference on Learning Representations, 2024.





Xueyao Wan and Hang Yu. Mmgraphrag: Bridging vision and language with interpretable multimodalknowledge graphs. arXiv preprint arXiv:2507.20804, 2025.





Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu,Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document contentextraction. arXiv preprint arXiv:2409.18839, 2024.





Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. Archrag: Attributed community-based hierarchical retrieval-augmented generation. arXiv preprint arXiv:2502.09891, 2025.





Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,Xu Han, Zhiyuan Liu, and Maosong Sun. Visrag: Vision-based retrieval-augmented generation onmulti-modality documents. arXiv preprint arXiv:2410.10594, 2025.





Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen,Yilin Xiao, Chuang Zhou, Yi Chang, and Xiao Huang. A survey of graph retrieval-augmentedgeneration for customized large language models. arXiv preprint arXiv:2501.13958, 2025.





Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, andDong Yu. Docbench: A benchmark for evaluating llm-based document reading systems. arXivpreprint arXiv:2407.10701, 2024.



# A APPENDIX

This appendix provides comprehensive supporting materials for our experimental evaluation andimplementation details. Section A.1 presents detailed dataset statistics for the DocBench andMMLongBench multi-modal benchmarks, including document type distributions and complexitymetrics. Section A.2 showcases additional case studies that demonstrate RAG-Anything’s structure-aware capabilities across diverse multimodal content understanding tasks. Section A.3 documents thecomplete set of multimodal analysis prompts for vision, table, and equation processing that enablecontext-aware interpretation. Section A.4 provides the standardized accuracy evaluation prompt usedfor consistent response assessment across all experimental conditions.

# A.1 DATASET CHARACTERISTICS AND STATISTICS


Table 5: Document type distribution and statistics for the DocBench benchmark.


<table><tr><td>Type</td><td>Acad.</td><td>Fin.</td><td>Gov.</td><td>Law.</td><td>News</td></tr><tr><td># Docs</td><td>49</td><td>40</td><td>44</td><td>46</td><td>50</td></tr><tr><td># Questions</td><td>303</td><td>288</td><td>148</td><td>191</td><td>172</td></tr><tr><td>Avg. Pages</td><td>11</td><td>192</td><td>69</td><td>58</td><td>1</td></tr></table>


Table 6: Document type distribution and statistics for the MMLongBench benchmark.


<table><tr><td>Type</td><td>Res.</td><td>Tut.</td><td>Acad.</td><td>Guid.</td><td>Broch.</td><td>Admin.</td><td>Fin.</td></tr><tr><td># Docs</td><td>34</td><td>17</td><td>26</td><td>22</td><td>15</td><td>10</td><td>11</td></tr><tr><td># Questions</td><td>292</td><td>138</td><td>199</td><td>155</td><td>100</td><td>81</td><td>117</td></tr><tr><td>Avg. Pages</td><td>39</td><td>58</td><td>35</td><td>78</td><td>30</td><td>17</td><td>87</td></tr></table>

Tables 5 and 6 present the distribution of document types across the DocBench and MMLong-Bench benchmarks. • DocBench encompasses medium- to long-length documents spanning variousdomains, including legal, governmental, and financial files. Financial reports represent the mostextensive category, averaging 192 pages per document, while the News category consists of concisesingle-page newspapers. • MMLongBench demonstrates a broader spectrum of document types andlengths. Research reports, tutorials, and academic papers maintain moderate lengths of 35–58 pageson average, while guidebooks extend to approximately 78 pages. Brochures and administrative filesremain relatively compact, whereas financial reports again emerge as the longest category.

Collectively, these two benchmarks provide comprehensive coverage ranging from brief news arti-cles to extensive technical and financial documentation. This establishes diverse and challengingevaluation contexts for multimodal document understanding tasks.

# A.2 ADDITIONAL CASE STUDIES

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/cfa0548ca603857a89bc882628fb87b65132d8a0a78cae8bba5a64e041d34f60.jpg)


GPT-4o-mini??:The removal of the dual co-attention mechanism from theGCAN sub-model resulted in thelowest accuracy for Twitter15.

MMGraphRAG??:The removal of the source tweet from the GCANmodel resulted in the lowest accuracy forTwitter15, as indicated by significant accuracydrops in the ablation analysis in Figure 4.

LightRAG??：Removing the source tweet from the GCAN modelresulted in the lowest accuracy for Twitter15, asindicated by a significant drop in performance whenusing the sub-model that excluded both source tweetembeddings and dual co-attention.

RAG-Anything(Correct??):The removal of the source tweet embeddingsand dual co-attention, indicated as modelconfiguration " -S-A, " resulted in the lowestaccuracy for Twitter15.


Figure 5: Visual reasoning case. RAG-Anything correctly identifies "-S-A" as the lowest accuracyconfiguration, while baselines misinterpret spatial relationships.


• Demonstrating Visual Reasoning Capabilities. Figure 5 illustrates how RAG-Anything handlescomplex visual reasoning tasks involving chart interpretation. The query asks which GCAN sub-model component removal yields the lowest accuracy on Twitter15. Traditional approaches struggle

with spatial relationships between visual elements. RAG-Anything addresses this challenge byconstructing a structured graph representation of the bar plot. Bars, axis labels, and legends becomeinterconnected nodes. These are linked by semantic relations such as bar-of and label-applies-to.

This graph-based approach enables precise alignment between visual and textual elements. Thesystem correctly identifies the bar labeled "-S-A" (removing source tweet embeddings and dualco-attention) and its corresponding accuracy value as the lowest performer. Baseline methods thatflatten visual information often misinterpret spatial relationships. They frequently conflate nearbycomponents. RAG-Anything’s structured representation preserves critical visual-textual associations.This leads to accurate query resolution and proper attribution of performance drops to "-S-A".

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/d53d5f010b3509d6f1e8c9e0c888f94af69668846221e2a3fbd92e38ee0a8e1e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/6c0bd9fbf06a20153dcb8c3dbe74800147dab2e5473394b918442bece356fcfe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/8b5d650838a0be291a112adb1f94ae0fa96c354b14af0052b7c8c34d04b3bdbb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/49f670bf5d7dcfd4a94f7042d1716e53453712f5b3c1ba4da921d9d045960de1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/9fa77e628c8067bd981cc79c269ff7e69ec7dabba1b578b2c7df81976a9482f4.jpg)



Figure 6: Tabular navigation case. RAG-Anything locates the highest AUPRC value (0.506), whilethe compared approaches struggle with structural ambiguity.


• Handling Complex Tabular Structures. Figure 6 showcases RAG-Anything’s ability to navigateintricate tabular data where structural disambiguation is crucial. The query seeks the model combi-nation achieving the highest AUPRC value for the Evidence Inference dataset—a task complicatedby repeated row labels across multiple datasets within the same table. This scenario highlights afundamental limitation of conventional approaches that struggle with structural ambiguity in data.

RAG-Anything overcomes this by parsing the table into a comprehensive relational graph whereheaders and data cells become nodes connected through explicit row-of and column-of relationships.This structured representation enables the system to correctly isolate the Evidence Inference datasetcontext and identify " $\mathrm { " G l o V e + L S T M - }$ Attention" with a score of 0.506 as the optimal configuration.By explicitly preserving hierarchical table constraints that other methods often collapse or misinterpret,RAG-Anything ensures reliable reasoning across complex multi-dataset tabular structures.

# A.3 CONTEXT-AWARE MULTIMODAL PROMPTING

These three prompts orchestrate structured, context-aware multimodal analysis with JSON-formattedoutputs. They systematically guide the model to extract comprehensive descriptions of visual, tabular,and mathematical content while maintaining explicit alignment with surrounding information.

Vision Analysis Prompt. Figure 7 orchestrates comprehensive image-context integration. Theprompt directs the model to systematically capture compositional elements, object relationships,visual attributes, stylistic features, dynamic actions, and technical components (e.g., charts), while es-tablishing explicit connections to accompanying text. This approach transcends superficial description,enabling contextually-grounded interpretations that enhance knowledge retrieval and substantiation.

Table Analysis Prompt. Figure 8 structures systematic tabular content decomposition across multipleanalytical dimensions: structural organization, column semantics, critical values, statistical patterns,and contextual relevance. Through precise terminology and numerical accuracy requirements, theprompt eliminates ambiguous generalizations and ensures faithful preservation of key indicatorswhile maintaining coherent alignment with surrounding discourse.

Equation Analysis Prompt. Figure 9 prioritizes semantic interpretation over syntactic restatementof mathematical expressions. The prompt instructs comprehensive analysis of variable definitions,operational logic, theoretical foundations, inter-formula relationships, and practical applications. Thismethodology ensures mathematical content becomes integral to broader argumentative frameworks,supporting enhanced retrieval accuracy, analytical traceability, and reasoning coherence.


Vision Analysis Prompt


```txt
1 Please analyze this image in detail, considering the surrounding context. Provide a JSON response with the   
2 following structure:   
3   
4 {   
5 "detailed_description": "A comprehensive and detailed visual description of the image following these   
6 guidelines:   
7 - Describe the overall composition and layout   
8 - Identify all objects, people, text, and visual elements   
9 - Explain relationships between elements and how they relate to the surrounding context   
10 - Note colors, lighting, and visual style   
11 - Describe any actions or activities shown   
12 - Include technical details if relevant (charts, diagrams, etc.)   
13 - Reference connections to the surrounding content when relevant   
14 - Always use specific names instead of pronouns",   
15 "entity_info": {   
16 "entity_name": {"entity_name"},   
17 "entity_type": "image",   
18 "summary": "concise summary of the image content, its significance, and relationship to surrounding content   
19 (max 100 words)"   
20 }   
21 }   
22   
23 Context from surrounding content:   
24 {context}   
25   
26 Image details:   
27 - Image Path: {image_path}   
28 - Captions: {captions}   
29 - Footnotes: {footnotes}   
30   
31 Focus on providing accurate, detailed visual analysis that incorporates the context and would be useful for   
32 knowledge retrieval.
```

Figure 7: Vision analysis prompt for context-aware image interpretation and knowledge extraction.


Table Analysis Prompt


```txt
table_analysis_prompt.png
1 Please analyze this table content considering the surrounding context, and provide a JSON response with the following structure:
2 following structure:
3
4 {
5 "detailed_description": "A comprehensive analysis of the table including:
6 - Table structure and organization
7 - Column headers and their meanings
8 - Key data points and patterns
9 - Statistical insights and trends
10 - Relationships between data elements
11 - Significance of the data presented in relation to surrounding context
12 - How the table supports or illustrates concepts from the surrounding content
13 Always use specific names and values instead of general references.", 
14 "entity_info": {
15 "entity_name": "{entity_name}", 
16 "entity_type": "table",
17 "summary": "concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)"
18 }
19 }
20 }
21 }
22 Context from surrounding content:
23 {context}
24 }
25 Table Information:
26 Image Path: {table_img_path}
27 Caption: {table_caption}
28 Body: {table_body}
29 Footnotes: {table_footnote}
30
31 Focus on extracting meaningful insights and relationships from the tabular data in the context of the surrounding content.
```

Figure 8: Table analysis prompt for structured content decomposition and semantic understanding.


Equation Analysis Prompt


```txt
equation_analysis_prompt.png   
1 Please analyze this mathematical equation considering the surrounding context, and provide a JSON response   
2 with the following structure:   
3   
4 {   
5 "detailed_description": "A comprehensive analysis of the equation including:   
6 - Mathematical meaning and interpretation   
7 - Variables and their definitions in the context of surrounding content   
8 - Mathematical operations and functions used   
9 - Application domain and context based on surrounding material   
10 - Physical or theoretical significance   
11 - Relationship to other mathematical concepts mentioned in the context   
12 - Practical applications or use cases   
13 - How the equation relates to the broader discussion or framework   
14 Always use specific mathematical terminology.",   
15 "entity_info": {   
16 "entity_name": "{entity_name}",   
17 "entity_type": "equation",   
18 "summary": "concise summary of the equation's purpose, significance, and role in the surrounding context (max   
19 100 words)"   
20 }   
21 }   
22   
Context from surrounding content:   
24 {context}   
25   
Equation Information:   
27 Equation: {equation_text}   
28 Format: {equation_format}   
29   
Focus on providing mathematical insights and explaining the equation's significance within the broader   
31 context.
```

Figure 9: Equation analysis prompt for mathematical expression interpretation and integration.


Accuracy Evaluation Prompt


```txt
accuracy_evaluation_prompt.png   
1 You are an expert evaluator tasked with assessing the accuracy of answers generated by a RAG   
2 (Retrieval-Augmented Generation) system.   
3   
4 \*\*Task\*: Evaluate whether the generated answer correctly responds to the given question based on the expected   
5 answer.   
6   
7 \*\*Question\*: {question}   
8   
9 \*\*Expected Answer\*: {expected_answer}   
10   
11 \*\*Generated Answer\*: {generated_answer}   
12   
13   
14 \*\*Evaluation Criteria\*:   
15 1. \*\*Accuracy (0 or 1)\*: Does the generated answer match the factual content of the expected answer?   
16 - 1: The generated answer is factually correct and aligns with the expected answer   
17 - 0: The generated answer is factually incorrect or contradicts the expected answer   
18   
19 \*\*Instructions\*:   
20 - Focus on factual correctness, not writing style or format   
21 - Consider partial matches: if the generated answer contains the correct information but includes additional   
22 context, it should still be considered accurate   
23 - For numerical answers, check if the values match or are equivalent   
24 - For list answers, check if all key elements are present   
25 - If the expected answer is "Not answerable" and the generated answer indicates inability to answer, consider   
26 it accurate   
27   
28 \*\*Output Format\*:   
29 Please respond with a JSON object containing only:   
30 {   
31 "accuracy": 0 or 1,   
32 "reasoning": "Brief explanation of your evaluation"   
33 }
```

Figure 10: Accuracy evaluation prompt for consistent factual assessment across question types.

# A.4 ACCURACY EVALUATION PROMPT DESIGN

Figure 10 presents the standardized prompt specifically designed for systematic factual accuracy as-sessment of generated responses across multiple domains. The prompt establishes explicit evaluationcriteria that prioritize content correctness over stylistic considerations, producing binary accuracy

classifications accompanied by concise analytical justifications. All accuracy evaluations throughoutour comprehensive experimental framework were conducted using GPT-4o-mini, ensuring consistentand reliable assessment standards across diverse question categories and specialized domains.

# A.5 CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG

While current multimodal RAG systems demonstrate promising capabilities, their limitations emergemost clearly through systematic analysis of failure cases. Understanding where and why these systemsbreak down is crucial for advancing the field beyond current performance plateaus. Examining failurepatterns helps identify fundamental architectural bottlenecks and design principles for more robustmultimodal systems. Our investigation reveals two critical failure patterns exposing deeper systemicissues in multimodal RAG architectures. These patterns are not merely edge cases but reflectfundamental challenges in cross-modal information integration and structural reasoning:

• Text-Centric Retrieval Bias: Systems exhibit strong preference for textual sources, even whenqueries explicitly demand visual information. This reveals inadequate cross-modal attention.

• Document Structure Processing Challenges: Systems struggle with complex layouts and non-linear information flows. This exposes limitations in spatial reasoning and contextual understanding.

These failure modes illuminate key insights about current multimodal AI. They provide concretedirections for architectural innovations that could substantially improve system robustness.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/2662a2f4905577d7008213b838cf2d581b9314741c3a427344d3a573e7e6dd62.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/a1eddd099be8c89a9af58a85909b9bdbe17781731b1ad431b6471b243fe79a34.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/44bd605f5c7c20b07569ddb12093ec4efb91635f589511657f56d4777996c2de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/08a20d1b8a4423b84204e9918992c60e371320ee6cb4dd98129e2f96fc66ce1c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/ae0d073fec8e9fc5912602c472ad1d217648e1c9e8a1305093aea7dadfed02bf.jpg)



Figure 11: Cross-modal noise case. All methods fail to retrieve the correct answer from the specifiedimage, instead retrieving noisy textual evidence that misaligns with the structured visual content.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/08776b90c3bf0686a0b14489df82d4a28ed1d2c697cc2fa65daf3d62b10e32b6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/3a2a789e40e2e34f5061cbad9d0e9d34fe10a4e8861595dbc883ceeb1d588b04.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/e463aea5b04288ae18185bd159ad748512f9137f6d84994db730778cc5eaa5c8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/9426e9c961e50d8baa2f4e8a63b0f2ff4f2d082a920320a7d624a0764fa14385.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-18/d0a56d4a-34e1-4849-a814-5badbb84cd27/60b72bbba99fbb664cc00f82b739910746c362c9a9d7c65a7dbffe5d1a69520b.jpg)



Figure 12: Ambiguous table structure case. All methods fail to correctly parse the confusing tablelayout with merged cells and unclear column boundaries, leading to incorrect data extraction.


Case 1: Cross-Modal Misalignment. Figure 11 presents a particularly revealing failure scenariowhere all evaluated methods consistently produce incorrect answers despite having access to thenecessary information. This universal failure across different architectures suggests fundamentallimitations in how current systems handle noisy, heterogeneous multimodal data—a critical challengeas real-world applications inevitably involve imperfect, inconsistent information sources. The failureexposes two interconnected systemic issues that compound each other:

Issue 1: Retrieval Bias Toward Text. Current RAG systems demonstrate pronounced bias towardtextual passages. This occurs particularly when visual content lacks exact keyword matches. Thebias persists even when queries contain explicit instructions to prioritize visual sources. This revealsa fundamental weakness in cross-modal attention mechanisms.

The retrieved textual information, while topically related, often operates at a different granularity levelthan visual content. Images may contain precise, structured data such as specific numerical values,

detailed diagrams, or exact spatial relationships. Corresponding text typically provides general,conceptual descriptions. This semantic misalignment introduces noise that actively misleads thereasoning process. The system attempts to reconcile incompatible levels of detail and specificity.

Issue 2: Rigid Spatial Processing Patterns. Current visual processing models exhibit fundamentalrigidity in spatial interpretation. Most systems default to sequential scanning patterns—top-to-bottom and left-to-right—that mirror natural reading conventions. While effective for simple textdocuments, this approach creates systematic failures with structurally complex real-world content.Many documents require non-conventional processing strategies. Tables demand column-wiseinterpretation, technical diagrams follow specific directional flows, and scientific figures embedcritical information in unexpectedly positioned annotations. These structural variations are prevalentin professional documents, making adaptive spatial reasoning essential.

In the observed failure case, the correct answer required integrating visual elements in reverse orderfrom the model’s default processing sequence. The system’s inability to recognize and adapt to thisstructural requirement led to systematic misinterpretation. This represents a fundamental architecturallimitation where spatial reasoning remains static regardless of document context or query intent.When spatial processing patterns are misaligned with document structure, the extracted informationbecomes not merely incomplete but actively misleading. This structural noise compounds otherprocessing errors and can lead to confident but entirely incorrect conclusions.

Case 2: Structural Noise in Ambiguous Table Layouts. As shown in Figure 12, all methods failedwhen confronted with a structurally ambiguous table. The primary failure stems from the table’sconfusing design: the GEM row lacks dedicated cell boundaries, and the "Joint" and "Slot" columnsmerge without clear separation. These structural irregularities create parsing ambiguities that system-atically mislead extraction algorithms. This failure pattern reveals a critical vulnerability in currentRAG systems. When table structures deviate from standard formatting conventions—through mergedcells, unclear boundaries, or non-standard layouts—extraction methods consistently misinterpret cellrelationships and conflate distinct data values. This exposes the brittleness of current approacheswhen faced with real-world document variations that deviate from clean, structured formats.

The case highlights two essential directions for enhancing robustness. RAG systems require layout-aware parsing mechanisms that can recognize and adapt to structural irregularities rather thanimposing rigid formatting assumptions. Additionally, integrating visual processing capabilitiescould significantly improve noise resilience, as visual models can leverage spatial relationships andcontextual design cues that are lost in purely structural representations.