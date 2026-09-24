# Destiny 2와 Tiger 엔진의 렌더링 파이프라인

## 연구 범위와 증거의 신뢰도

Tiger의 실제 렌더러는 공개 소스가 아니므로, 외부에서 작성할 수 있는 “완전한 파이프라인”은 엔진 코드 자체의 재현이 아니라 여러 시기의 Bungie 공식 발표, Bungie.net용 3D 자산 명세, 기술 아티스트의 설명, 공개된 웹 렌더러, 그리고 리버스 엔지니어링 프로젝트를 교차 검증해 구성한 **가장 근거가 강한 재구성**입니다. 핵심 자료는 2013년 Destiny 렌더링 발표, 2014년 Gear 제작 발표, 2015년 Tiger 엔진·멀티스레딩 발표, 2017년 TFX와 파티클 시스템 발표, 2018년 Destiny 2의 PBR·IBL 발표, 2019년 Bungie.net 3D Content Documentation, 그리고 2020년 이후 엔진 및 콘텐츠 파이프라인 갱신 기록입니다. Tiger는 2008년 중반부터 개발됐으며, Reach 계열 기술에서 출발하되 멀티스레딩·크로스플랫폼·게임 로직과 엔진 기능의 분리를 핵심 원칙으로 삼아 재구축된 엔진입니다. citeturn18view4turn19view0turn19view1

자료의 신뢰도는 세 층으로 나눠야 합니다. 첫째는 Bungie 또는 발표자가 직접 설명한 **확정 사실**, 둘째는 공식 자산 메타데이터와 여러 구현체가 일치해 지지하는 **강한 복원**, 셋째는 이름과 데이터 흐름은 확인되지만 정확한 수식이나 비트 배치가 공개되지 않은 **추정 영역**입니다. 특히 2013년 자료는 초대 Destiny의 파이프라인이고, 2018년 자료는 Destiny 2 출시기의 구조이며, 2020년에는 콘텐츠 빌드·캐릭터 얼굴·런타임 데칼·일부 지역 조명 등이 다시 갱신되었습니다. 그러므로 아래 글은 특정 버전의 한 프레임 덤프라기보다 **Tiger 렌더러의 계보와 Destiny 2에서 확인되는 현대적 형태**를 함께 설명합니다. citeturn16view0turn16view1

또한 Bungie.net의 3D 콘텐츠와 Spasm은 게임 렌더러의 완전한 공개판이 아닙니다. 공식 문서 자체가 Bungie.net에 제공되는 자료를 “아직 포괄적이지 않은 modest collection”으로 표현하며, 투명 셰이더 등 여러 런타임 효과가 웹 뷰어에서 지원되지 않는다고 명시합니다. lowlidev가 분석한 Spasm 역시 무기와 방어구를 웹·Companion 환경에서 미리 보여주기 위한 별도 WebGL 라이브러리이며, Destiny 2의 PBR 데이터를 온전히 재현하지 못한 부분 구현입니다. 따라서 Spasm의 셰이더 코드를 게임의 원본 HLSL로 간주해서는 안 됩니다. citeturn23view2turn5view1turn6view5

Marathon과의 관계도 같은 주의가 필요합니다. Bungie의 Tiger 공유 조직에 관한 공개 채용공고 인용은 Destiny, Marathon과 다른 프로젝트 사이에서 도구·엔진 업그레이드를 더 쉽게 공유하려는 의도를 보여주며, Marathon에 참여한 개발자의 공개 경력에도 Tiger Engine이 명시되어 있습니다. 그러나 이는 Marathon이 Destiny 2의 G-buffer 배치, 조명 수식, 투명 패스, Gear 셰이더를 비트 단위로 동일하게 사용한다는 증거는 아닙니다. 가장 안전한 결론은 **공통 Tiger 기반과 상당한 도구·자원·셰이더 시스템의 혈통은 확실하지만, Marathon은 별도 분기에서 재질 ABI와 렌더 패스를 확장하거나 교체했을 가능성이 높다**는 것입니다. citeturn14search3turn14search12

이 연구에서 복원되는 전체 구조는 다음과 같습니다.

```text
원본 DCC 자산·셰이더 컴포넌트·게임플레이 속성
        ↓ 콘텐츠 import / validation / preprocessing
Tiger 태그·리소스·Technique·TFX·플레이트 텍스처
        ↓ 패키지 스트리밍
가시성 DB·LOD·오클루전·인스턴스 선택
        ↓ 멀티스레드 render jobs / draw packet 생성
조건부 depth prepass · shadow generation
        ↓
불투명 geometry → 압축 G-buffer
        ↓
일반 decal → investment decal → 기타 G-buffer 수정
        ↓
직접광 · 그림자 · IBL · cubemap · 간접광
        ↓
additive decal · forward/transparent geometry
        ↓
파티클 · 볼류메트릭 효과 · 대기 · light shafts
        ↓
노출 · 색보정 · HDR 변환 · AA · 최종 합성
```

여기서 중요한 점은 Bungie 자산의 `render stage` 숫자가 반드시 GPU command encoder의 시간 순서를 의미하지는 않는다는 것입니다. 공식 명세에서 `generate gbuffer`는 stage 0이고 `depth prepass`는 stage 12인데, 의미상 depth prepass는 G-buffer보다 먼저 실행될 수 있습니다. 따라서 이 값들은 단순한 프레임 순번이라기보다 geometry part가 참여할 **렌더링 범주 또는 ABI 식별자**로 해석하는 편이 타당합니다. 이는 공개 명칭에서 도출되는 강한 추론입니다. citeturn23view2

## 프레임을 만드는 엔진 구조

Tiger의 프레임은 렌더 스레드 하나가 모든 오브젝트를 순회하고 즉시 draw call을 발행하는 전통적 구조와 거리가 있습니다. Bungie는 엔진 대부분을 세밀한 작업 그래프로 만들고, 메인 게임 루프의 일부를 `job fibers`라고 부르는 작업 열로 분해했습니다. 런타임 리소스 추적 시스템은 프레임 중 데이터 읽기·쓰기·생성·삭제가 올바른 시점에 이루어지는지 검증하며, 시스템 단위 파이프라이닝과 데이터 병렬화를 통해 시뮬레이션에서 GPU 제출까지 연결합니다. citeturn18view3

이 구조는 렌더링에서 특히 중요합니다. 가시성 계산, LOD 결정, 셰이더 속성 평가, 상수 버퍼 준비, draw packet 생성과 정렬, GPU 리소스 준비가 서로 다른 작업으로 분리될 수 있기 때문입니다. 2015년 Bungie 발표에서는 이전 세대와 새 세대 콘솔 사이에서 CPU 제출 예산을 크게 늘리지 않으면서도 거의 한 자릿수 배수에 가까운 draw-call 증가와 더 많은 G-buffer 요소, 그림자, 효과, 1080p 출력을 처리해야 했다고 설명합니다. citeturn8view3turn16view3

**가시성의 첫 단계**에는 Umbra 기반 공간 데이터가 사용됐습니다. Destiny의 Umbra 통합은 아티스트가 밀폐되고 엄격한 BSP형 월드를 만들지 않고도 비교적 느슨한 polygon soup 형태로 환경을 제작하게 하고, import 과정에서 자동 공간 데이터베이스를 구축하는 방식이었습니다. 이 데이터베이스는 오클루전 컬링뿐 아니라 공간 추론, 경로 탐색, 오디오 전달, 조명 관련 질의에도 활용할 수 있도록 설계됐습니다. citeturn23view0

그 결과 프레임의 geometry 집합은 대략 다음 조건을 거쳐 좁혀집니다.

```text
현재 로드된 destination cell / resource set
→ camera frustum에 들어오는 공간과 오브젝트
→ Umbra visibility 및 occlusion 결과
→ 개별 인스턴스와 geometry component
→ 거리·화면 크기·플랫폼 예산에 맞는 LOD
→ 해당 LOD에서 활성화되는 render-stage part
→ material/technique별 draw packet
```

Bungie.net 명세에는 LOD 0부터 3까지의 단순 단계 외에도 `0`, `01`, `012`, `0123`, `1`, `12`, `123`, `2`, `23`, `3`처럼 어떤 상세 단계에서 part를 활성화할지를 나타내는 범주가 존재합니다. 예를 들어 `012` part는 LOD 0·1·2에서 사용되고 최저 상세 단계에서는 빠집니다. 이는 Quicktag가 단순히 가장 높은 LOD ID 하나만 택하는 대신, 현재 목표 LOD를 포함하는 모든 part 범주를 해석해야 한다는 뜻입니다. citeturn23view2

**렌더 스테이지 분류**는 공개 명세에서 최소한 다음 범주가 확인됩니다.

| 공개된 stage | 의미 |
|---|---|
| Generate G-buffer | 대부분의 불투명 geometry |
| Decals | 일반 데칼 |
| Investment decals | 장비·외형 시스템이 배치하는 별도 데칼 |
| Shadow generate | 그림자 맵 생성용 geometry |
| Decals additive | 가산 합성 데칼 |
| Transparents | 복잡한 투명·특수 셰이더 geometry |
| Depth prepass | 깊이 전용 사전 패스 |
| Redacted stages | 게임 전용이지만 공개 웹 자료에서는 명칭이 제거된 패스 |

총 stage 수는 19개로 명시돼 있으므로, 웹에서 이름이 보이는 몇 패스만으로 실제 Tiger 프레임 전체를 설명할 수는 없습니다. 숨겨진 stage에는 속도 벡터, 특정 조명 전용 geometry, distortion, 특수 depth, 볼류메트릭 또는 플랫폼별 기능이 포함될 가능성이 있지만, 공개 자료만으로 각각을 단정할 수는 없습니다. citeturn23view2

**재질과 Technique를 draw packet으로 바꾸는 핵심 계층이 TFX**입니다. TFX 컴포넌트는 편집 가능한 속성, HLSL 코드, GPU state를 한 단위로 묶고, expression을 통해 콘텐츠 속성과 엔진 런타임 상태를 GPU 상태에 연결합니다. 메타데이터는 아티스트용 UI를 생성하는 것에 그치지 않고 import 과정의 데이터 전처리, 필요한 vertex stream의 자동 선택, alpha-test 같은 가시성 특성 판정, draw-call 제출 데이터 최적화, 멀티스레드 렌더 작업의 부하 분산에도 사용됩니다. citeturn3view1turn22search0

TFX expression은 단순한 오프라인 셰이더 노드가 아닙니다. Bungie의 설명에 따르면 visual function editor에서 만든 동적·애니메이션 expression은 import 시 bytecode로 컴파일되고, 런타임에는 CPU 또는 GPU에서 해석될 수 있습니다. 다른 경로에서는 expression을 HLSL로 변환할 수도 있습니다. 또한 custom scope를 통해 프레임, 뷰, 오브젝트, 장비와 같이 서로 다른 갱신 주기를 가진 데이터를 분리하며, 패치 가능한 셰이더 패키지와 오프라인 컴파일된 셰이더 사이의 인터페이스도 유지합니다. citeturn3view1turn23view5

이 때문에 Tiger의 “Technique”는 일반적인 `vertex shader + pixel shader + texture list`보다 넓은 개념입니다. 실제로는 다음을 함께 정의하는 렌더링 계약에 가깝습니다.

```text
GPU shader code와 permutation
vertex input 요구사항
render target / depth / blend / rasterizer state
texture와 sampler binding
inline constant와 resource scope
object·view·frame runtime expression
alpha test와 visibility 조건
import-time preprocessing
runtime draw submission metadata
```

Quicktag가 shader tag에서 texture hash만 찾아 하나의 범용 WGSL 재질에 대응시키면 색과 형태는 어느 정도 보일 수 있지만, TFX가 결정하는 분기·scope·state·오브젝트 채널·pass 참여 조건은 손실됩니다. 실제 Tiger와의 가장 큰 구조적 차이는 바로 이 지점입니다. citeturn22search0turn3view1

## 불투명 지오메트리와 조명 파이프라인

초대 Destiny의 공개 렌더링 자료는 기본 골격을 **deferred renderer**로 설명합니다. 먼저 geometry를 렌더링하여 G-buffer를 만들고, 이후 광원의 영향 영역을 나타내는 geometry 또는 volume을 그리면서 diffuse와 specular 조명을 누적합니다. Bungie는 이때 제한된 메모리와 대역폭에서 다양한 재질을 표현하기 위해 압축되고 유연한 G-buffer 표현을 강조했습니다. citeturn2search8turn20search11

Destiny 2에서는 이 구조를 폐기했다기보다, 재질과 조명 모델을 **physically inspired material model 및 image-based lighting pipeline**으로 전환하면서 G-buffer를 다시 최적화한 것으로 설명됩니다. 2018년 Bungie 발표는 재질, 조명 기능, 데칼을 PBR 파이프라인의 확장 사례로 다루고, 콘텐츠의 물리적 타당성 검증과 게임플레이 가독성을 위한 최종 프레임의 아트 디렉션을 함께 논의합니다. 즉 Destiny 2는 엄격한 오프라인 물리 시뮬레이터라기보다, 물리 기반 변수와 에너지 관계를 공통 토대로 삼되 필요하면 연출과 가독성을 위해 확장하는 **physically inspired deferred/IBL renderer**입니다. citeturn16view1

공개 자료가 현재 Destiny 2의 정확한 render-target 포맷과 비트 패킹을 제공하지는 않지만, 자산 메타데이터와 디버그 뷰에서 최소한 다음 논리적 surface 정보가 재구성됩니다.

```text
base albedo / tint
surface normal과 detail normal
roughness 또는 smoothness
metalness
ambient occlusion
emissive와 intensity
alpha test / fringe / transparency
dye selection 및 dye strength
wear와 worn-material 선택
subsurface-scattering 관련 값
decal 및 material-family별 추가 파라미터
```

이 값들이 모두 독립적인 풀 해상도 G-buffer 채널로 저장된다는 의미는 아닙니다. 일부는 geometry pass에서 최종 albedo·normal·roughness로 합성되고, 일부는 압축·양자화되어 저장되며, 일부 특수 재질은 별도 forward stage에서만 사용될 수 있습니다. 공식 발표가 “compressed and flexible G-buffer”와 G-buffer 최적화를 강조하는 만큼, Tiger가 논리적 material property 수보다 적은 물리적 채널로 정보를 압축한다고 보는 것이 합리적입니다. citeturn20search11turn16view1

**일반적인 불투명 프레임의 가장 강한 복원**은 다음 순서입니다.

```text
선택적 depth-only prepass
→ 주요 광원용 shadow-map generation
→ opaque geometry의 G-buffer 작성
→ alpha-tested opaque geometry 작성
→ 일반 decal의 G-buffer 수정
→ investment decal의 G-buffer 수정
→ 다른 공개되지 않은 material-resolve 단계
→ light volume 또는 screen-space deferred direct lighting
→ shadow 적용
→ image-based diffuse/specular lighting
→ atmosphere·fog와 결합할 scene lighting 결과 생성
```

Depth prepass가 모든 오브젝트에 항상 사용되는 것은 아닙니다. Bungie.net 문서도 이를 “available, but not always necessary”라고 표현합니다. 따라서 Technique, 플랫폼, 오버드로우 비용, alpha test, 이후 패스의 depth 요구에 따라 선택적으로 사용될 가능성이 높습니다. citeturn23view2

**Decal은 하나의 종류가 아닙니다.** 일반 decal, investment decal, additive decal이 별도 stage로 분리되어 있습니다. 일반 환경 데칼은 벽의 얼룩·표식·손상처럼 월드 제작자가 배치하는 표면 수정일 가능성이 높고, investment decal은 아이템 외형, 상징, 제조사 로고, 장비 장식 등 게임의 장비·커스터마이징 데이터가 결정하는 별도 계층입니다. Additive decal은 발광 문양이나 에너지 효과처럼 조명 결과 위에 가산적으로 놓이는 재질에 적합합니다. 앞의 두 설명은 stage 이름과 Gear 메타데이터를 결합한 해석이며, 세 stage가 독립적으로 존재한다는 사실 자체는 공식 문서로 확인됩니다. citeturn23view2turn23view3

**직접광**은 G-buffer의 normal과 material property를 읽고 diffuse와 specular 성분을 계산합니다. 2013년 설명은 이 두 성분을 별도로 누적한다고 명시하고, Destiny 2 발표는 이를 물리적으로 영감을 받은 material model로 옮겼다고 설명합니다. 다만 공개 자료만으로 GGX인지, 어떤 masking-shadowing term과 Fresnel 근사를 쓰는지, multi-scattering energy compensation을 사용하는지를 확정할 수 없습니다. Quicktag가 일반적인 GGX·Schlick 모델을 사용하는 것은 합리적인 시각적 근사이지만, 그것이 Tiger의 정확한 BRDF라는 증거는 아닙니다. citeturn2search8turn16view1

**간접광과 cubemap**은 Destiny 초기부터 재질 표현의 중요한 부분이었습니다. 2013년 자료는 cubemap이 specular reflection만 제공하는 고전적인 방식에 머무르지 않고 diffuse color와 specular 응답을 수정할 수 있다고 설명합니다. Destiny 2에서는 이를 본격적인 image-based lighting pipeline으로 전환했기 때문에, 로컬 reflection probe 또는 environment cubemap이 roughness에 따른 specular뿐 아니라 장면의 간접 diffuse 분위기와 재질 가독성에도 관여하는 구조로 보는 것이 타당합니다. citeturn21search1turn16view1

정확한 probe 구축 방식, prefiltered mip의 분포, BRDF integration LUT, probe blending과 parallax correction은 공개 자료에서 확인되지 않습니다. 따라서 Quicktag의 단일 cubemap과 단순 reflection vector 샘플링은 게임의 실제 IBL에 비해 상당히 축약된 모델일 가능성이 큽니다. 실제 월드 렌더러는 여러 공간 probe, 하늘 조명, 시간대 또는 지역별 조명 데이터, 아트 디렉션 보정치를 결합할 수 있습니다. 이는 D2 발표가 “image-based lighting pipeline”과 최종 프레임 아트 디렉션을 함께 강조한다는 사실에서 도출되는 추론입니다. citeturn16view1

**그림자 알고리즘은 가장 주의해야 하는 영역**입니다. `shadow generate` stage의 존재는 확정되지만, 현재 Destiny 2가 어떤 shadow map 필터를 쓰는지는 이번에 확인한 공개 자료에서 명시되지 않습니다. Bungie는 Halo 시대부터 Variance Shadow Maps와 Exponential Shadow Maps를 연구했으며, ESM의 단일 채널 저장, prefilter 가능성, 넓은 커널에서의 light bleeding 감소 등을 발표했습니다. 그러나 이것은 Bungie의 연구 계보이지, Destiny 2가 현재 ESM을 그대로 사용한다는 증거는 아닙니다. citeturn16view4turn17view3

**콘텐츠 검증과 디버그 뷰**도 렌더 파이프라인의 일부입니다. Bungie는 geometry budget, overdraw, light overdraw, texture 크기와 mip 사용, pixel cost, triangle 및 draw-call 비용을 시각화하는 개발 도구를 사용했습니다. Destiny 2 PBR 발표 역시 콘텐츠가 올바른 물리 범위 안에 있는지 검증하고 최종 프레임을 게임플레이 목적에 맞게 조정하는 워크플로를 핵심 주제로 삼았습니다. 즉 디버그 모드는 부가 기능이 아니라, 대규모 콘텐츠 팀이 재질 규약과 성능 예산을 유지하기 위한 시스템적 장치입니다. citeturn4view4turn16view1

## Gear 재질 시스템의 전체 해부

Destiny의 무기와 방어구는 “고유한 완성 텍스처 세 장을 각 아이템마다 준비하는” 방식으로 이해하면 안 됩니다. 기본 geometry와 마스크, 고정된 텍스처 영역, 교체 가능한 dye, detail texture, worn 상태, investment decal을 조합하여 수많은 외형을 만드는 **계층형 재질 합성 시스템**입니다. 이 시스템은 콘텐츠 중복과 메모리 사용을 줄이면서도 하나의 geometry에 여러 shader item과 장식 상태를 적용할 수 있도록 설계되었습니다. citeturn23view2turn23view3turn23view4

**Geometry part**에는 index 범위, primitive type, LOD 범주, shader 또는 Technique 참조와 함께 `gear_dye_change_color_index`가 존재합니다. 이 값은 해당 part가 몇 번째 dye slot을 사용할지, primary와 secondary 중 어느 material branch를 사용할지, investment decal이 적용되는 영역인지를 결정하는 lookup index로 쓰입니다. 공개된 웹 데이터의 lookup은 0·1이 dye slot 0, 2·3이 slot 1, 4·5가 slot 2, 6·7이 slot 3과 연관되며, 각 쌍은 primary/secondary 선택을 달리합니다. 6·7은 investment decal 사용과도 연결됩니다. citeturn23view3

이 구조는 geometry 자체가 “빨간 금속”이나 “흰 플라스틱”이라고 직접 지정하는 것이 아니라 다음과 같은 의미를 갖게 합니다.

```text
이 triangle range는 dye family N을 사용한다
이 range는 그 dye의 primary 또는 secondary branch를 사용한다
이 range는 locked/default/custom dye 교체 규칙의 영향을 받는다
이 range는 investment decal을 받을 수 있다
이 range의 packed control 값으로 worn·metal·alpha·emissive를 결정한다
```

### Texture plating

Gear 자산은 여러 작은 텍스처를 런타임에 무작위로 배치하지 않고, 각 geometry texture가 plate 안에서 차지할 위치를 미리 정해 두는 plating 시스템을 사용합니다. 한 plate set은 보통 diffuse, normal, gearstack 세 장으로 이루어지고, 각 하위 텍스처는 고정된 크기와 좌표에 배치됩니다. 고정 layout을 사용하면 여러 장비 조각에 필요한 작은 texture를 더 큰 atlas로 조립해 texture binding과 메모리 관리를 효율화할 수 있습니다. citeturn23view3

Bungie.net 웹용 콘텐츠는 실제 게임과 동일한 plating 동작을 완전히 구현하지 않았고, lowlidev 분석에 따르면 개별 하위 텍스처마다 plate 크기의 별도 이미지를 만드는 경우가 있었습니다. 이는 Spasm이나 Bungie.net 출력물을 분석할 때 관찰되는 비효율이 게임 엔진의 원래 atlas 전략과 동일하지 않을 수 있음을 뜻합니다. citeturn23view3

### Diffuse, normal, gearstack

공식 문서에서 Gear의 기본 texture set은 다음처럼 설명됩니다.

| 텍스처 | 주된 역할 |
|---|---|
| Diffuse | 종종 grayscale이며 dye albedo 색으로 착색되고 dye detail diffuse와 혼합 |
| Normal | 기본 표면 normal이며 dye detail normal과 혼합 |
| Gearstack | dye, 표면 속성, alpha·fringe 등 여러 정보를 채널에 포장한 control texture |

따라서 Gear의 diffuse를 그대로 최종 albedo로 표시하면 안 됩니다. 회색조 값은 표면의 명암·재료 디테일·오염 또는 패턴을 보존하는 기반이고, 최종 색은 선택된 dye tint 및 detail texture와 결합되어 만들어집니다. citeturn23view2

Destiny 2 gearstack 채널에 관해서는 lowlidev 작성자가 당시 Bungie Graphics Technical Art Lead였던 Nate Hawbaker의 설명을 기록했습니다. 그 복원에 따르면 red는 ambient occlusion, green은 smoothness, blue는 alpha-test와 emissive 관련 값을 인코딩하며, alpha에는 dye mask, 비염색 영역 metalness, wear mask 등이 범위 또는 규약에 따라 함께 인코딩됩니다. 특히 blue와 alpha는 단순한 단일 grayscale 의미가 아니라 여러 논리 상태가 압축된 채널이므로, 단순히 `B = emissive`, `A = dye`처럼 사용하는 것은 부정확합니다. citeturn6view2turn23view2

### Dye의 해석과 우선순위

공식 자료에는 세 dye group이 존재합니다.

- **Default dyes**는 shader item을 적용하지 않았을 때 장비가 사용하는 기본값입니다.
- **Custom dyes**는 장착한 shader 또는 외형 아이템이 제공하며 default를 대체합니다.
- **Locked dyes**는 shader 교체와 무관하게 장비가 유지하는 영역입니다.

lowlidev가 복원한 Spasm 로직에서도 default, custom, locked 순으로 slot을 resolve하며, custom은 shader gear에서 가져오고 locked는 원래 gear에서 다시 덮어씁니다. Destiny 1의 Exotic처럼 특정 부위를 항상 고유 색으로 유지하는 아이템에서 locked dye가 특히 흔했고, Destiny 2에서는 빈도가 줄었지만 시스템은 유지됩니다. citeturn23view2turn23view4

각 dye는 단순 RGB tint가 아니라 texture와 다수의 `vec4` material property를 가집니다. 공식적으로 확인되는 항목은 다음 계열입니다.

```text
detail_diffuse_transform
detail_normal_transform
spec_aa_xform

primary / secondary emissive tint and intensity bias
specular_properties
lobe_pbr_params
tint_pbr_params
emissive_pbr_params

primary / secondary albedo_tint
primary / secondary material_params
primary / secondary material_advanced_params
primary / secondary roughness_remap

primary / secondary worn_albedo_tint
primary / secondary wear_remap
primary / secondary worn_roughness_remap
primary / secondary worn_material_parameters

primary / secondary subsurface_scattering_strength_and_emissive
```

이 목록만으로도 Gear Dye가 색상 팔레트 이상의 시스템임을 알 수 있습니다. 하나의 dye는 primary·secondary 두 재질, 각각의 거칠기 변환, 마모된 상태의 별도 색과 표면 특성, detail diffuse·normal tiling, specular anti-aliasing 조정, emissive, subsurface 관련 응답까지 정의합니다. citeturn23view2turn20search5

### 추정 가능한 Gear fragment 흐름

정확한 Tiger HLSL은 공개되지 않았지만, 공식 속성과 마스크 의미를 결합하면 Gear surface resolve는 개념적으로 다음에 가깝습니다.

```text
plate diffuse, normal, gearstack 샘플
        ↓
geometry의 change-color index로 dye slot과 primary/secondary 선택
        ↓
default/custom/locked 규칙으로 최종 dye 결정
        ↓
grayscale diffuse × 선택된 albedo tint
        ↓
detail diffuse transform으로 미세 패턴 합성
        ↓
base normal + transformed detail normal 합성
        ↓
gearstack smoothness를 roughness remap에 통과
        ↓
gearstack mask로 dyed / non-dyed metal 응답 선택
        ↓
wear mask를 remap하여 pristine와 worn material 혼합
        ↓
alpha-test/fringe, emissive, AO 복원
        ↓
decal과 investment decal 적용
        ↓
G-buffer용 최종 albedo·normal·roughness·metal·AO·emissive 출력
```

이 순서는 메타데이터 관계에 기반한 복원이며, 실제 엔진은 일부 연산을 합치거나 G-buffer에 압축해 기록할 수 있습니다. 특히 `material_params`, `advanced_params`, `lobe_pbr_params`, `tint_pbr_params` 각 성분의 정확한 수학적 의미는 공식 문서에서도 “추후 공개” 상태로 남아 있습니다. citeturn23view2turn20search5

첨부하신 디버그 캡처는 이러한 분해를 시각적으로 잘 보여줍니다. 한 weapon material이 Dye, Worn Dye, Dye Detail, Ambient Occlusion, Roughness, Emission, Transparency, Metalness로 나뉘며, 별도 디버그 항목에는 Dye Mask와 Wear Mask가 보입니다.

![Destiny 계열 무기 재질의 dye, wear, AO, roughness, emission, transparency, metalness 디버그 레이어](sandbox:/mnt/data/203678ca-48e8-461b-af7c-ad4c75ae35d2.png)

이 이미지만으로 각 값의 저장 채널이나 수식을 확정할 수는 없지만, **최종 표면이 단일 albedo texture가 아니라 dye·wear·detail·표면 물성·투명·발광을 단계적으로 resolve한 결과**라는 점은 공식 Gear 메타데이터와 정확히 부합합니다. citeturn23view2turn6view2

### Worn material

Wear는 단순히 표면에 검은 얼룩을 곱하는 기능이 아닙니다. primary와 secondary 각각에 대해 worn albedo tint, wear remap, worn roughness remap, worn material parameters가 따로 존재합니다. 따라서 wear mask는 깨끗한 재질과 별도로 정의된 worn material 사이의 전환 인자로 쓰일 가능성이 높습니다. 마모된 도장 아래에서 금속이 드러나거나, 표면이 더 거칠어지거나 매끄러워지고, 색이 바뀌는 효과를 일관되게 만들 수 있습니다. citeturn23view2turn20search5

개념적으로는 다음과 같습니다.

```text
wear = saturate(inputWear × scale + bias)
cleanAlbedo  = base × cleanTint
wornAlbedo   = base × wornTint
finalAlbedo  = lerp(cleanAlbedo, wornAlbedo, wear)

cleanRoughness = remap(smoothness, cleanRoughnessParams)
wornRoughness  = remap(smoothness, wornRoughnessParams)
finalRoughness = lerp(cleanRoughness, wornRoughness, wear)

cleanMaterial ↔ wornMaterial 역시 wear로 전환
```

Quicktag에서 Weapon Mod Wear를 scratches·grime·damage texture의 tri-planar 합성으로 구현한 것은 Marathon의 특정 shader family를 복원하려는 별도 시도일 수 있습니다. 그러나 Destiny 2의 공개 Gear 자료가 보여주는 보편적 wear 모델은 geometry의 gearstack mask와 dye의 clean/worn material parameter를 결합하는 방식입니다. 두 체계는 공존할 수도 있으며, 모든 wear를 하나의 구현으로 통합해서는 안 됩니다.

### Roughness와 specular anti-aliasing

Gearstack은 smoothness를 제공하고 dye는 primary·secondary 및 worn branch별 roughness remap을 제공합니다. 이는 아티스트가 texture에 최종 roughness를 직접 굽는 대신, 공통 smoothness 신호를 재질별 범위로 변환할 수 있게 합니다. 같은 texture 신호를 플라스틱, 도장 금속, 천, 노출 금속에 다르게 해석할 수 있는 구조입니다. citeturn6view2turn23view2

`spec_aa_xform`의 존재는 normal-map과 미세 geometry가 만드는 고주파 specular aliasing을 재질 단위로 완화하는 처리가 있다는 강한 증거입니다. 다만 이 값이 Toksvig, LEAN mapping, normal variance 기반 roughness 보정 또는 Bungie 고유 방식 중 무엇에 정확히 대응하는지는 공개 자료만으로 확정할 수 없습니다. Quicktag가 이 값을 무시하면 작은 normal detail이 원거리에서 반짝이거나, authored roughness보다 표면이 지나치게 선명하게 보일 수 있습니다. citeturn23view2turn20search5

### Normal과 detail material

기본 normal은 geometry 고유의 큰 표면 형상을 제공하고, dye detail normal은 장착한 shader가 재질의 미세 질감을 바꾸는 데 사용됩니다. 예를 들어 동일한 방어구 geometry에 매끈한 도장, 섬유, 거친 분말 코팅 또는 새겨진 패턴을 서로 다른 detail normal과 transform으로 적용할 수 있습니다. `detail_diffuse_transform`과 `detail_normal_transform`이 분리되어 있으므로 두 detail texture는 서로 다른 scale과 offset을 사용할 수 있습니다. citeturn23view2

### Metalness, emissive, transparency와 subsurface

Gearstack의 packed mask 및 dye material property는 non-dyed metalness, emissive, alpha-test·fringe를 제어합니다. 이 구조는 금속 여부를 albedo 밝기만으로 추측하지 않고 authored control과 dye parameter로 결정한다는 뜻입니다. Quicktag처럼 texture가 없을 때 색과 밝기로 metalness를 추정하는 방식은 fallback으로는 유용하지만, 원본 Gear 렌더링을 목표로 할 때는 사용해서는 안 됩니다. citeturn6view2turn23view2

Primary와 secondary 각각에 subsurface-scattering strength와 emissive 관련 값도 존재합니다. 따라서 천, 피부 유사 소재, 반투명 합성수지, 에너지 재질처럼 일반적인 opaque metal/roughness 모델로 표현하기 어려운 표면을 위한 추가 lobe가 Gear shader에 포함될 수 있습니다. 정확히 screen-space SSS를 수행하는지, wrapped diffuse 또는 단순 transmission 근사를 사용하는지는 공개 메타데이터만으로 판단할 수 없습니다. citeturn23view2

## 투명체·VFX·대기·후처리

불투명 G-buffer 이후에는 일반적으로 G-buffer에 저장하기 어려운 효과가 별도 경로에서 처리됩니다. 공식 stage에는 additive decal과 transparents가 있고, 투명 part 대부분은 웹 뷰어가 지원하지 않는 복잡한 셰이더 효과를 사용한다고 명시돼 있습니다. 유리, 홀로그램, 에너지 막, 굴절, distortion, 입자, beam, 특수 emissive surface가 이 계층에 속할 가능성이 높습니다. citeturn23view2

**투명체는 deferred lighting의 단순 연장으로 처리하기 어렵습니다.** G-buffer는 화면 pixel당 보통 하나의 가장 가까운 surface를 나타내므로 여러 깊이의 투명 layer를 저장하기에 적합하지 않습니다. 따라서 투명 geometry는 불투명 장면의 depth와 조명 결과를 입력으로 사용하는 forward 또는 특수 composite pass에서 렌더링될 가능성이 큽니다. 공개 stage 분리가 이를 뒷받침하지만, 투명 조명이 clustered인지, tile light list를 사용하는지, 일부 빛을 별도 buffer에서 읽는지는 공개 자료에서 확인되지 않습니다. citeturn23view2turn2search8

초대 Destiny는 연기와 입자처럼 fill-rate가 큰 투명 효과를 낮은 해상도 buffer에 그리는 최적화를 사용했습니다. 단순한 4분의 1 해상도 렌더 후 bilinear upsample은 geometry 경계에서 입자가 새거나 끊기는 artifact를 만들었고, Bungie는 depth를 downsample하면서 최소·최대 깊이를 유지하고 입자 색과 alpha도 두 depth 경우에 대해 누적한 뒤, full-resolution depth를 이용해 min·max 결과 또는 둘의 혼합을 선택하는 개선된 합성법을 사용했습니다. citeturn8view3turn17view1

이 기법은 D1 시기의 공개 사례이므로 현재 Destiny 2의 모든 투명 효과가 정확히 같은 방식이라고 단정할 수는 없습니다. 그러나 고비용 파티클을 저해상도에서 처리하고 depth-aware reconstruction으로 불투명 경계를 보존한다는 설계 원칙은 대규모 VFX를 안정적인 프레임 시간 안에 넣기 위한 Tiger 계열의 중요한 전통입니다. citeturn16view3

**Destiny 2 파티클 시스템**은 node graph와 expression 중심입니다. 각 node의 particle size, color 등의 parameter 자체가 expression이며, expression-to-HLSL converter와 CPU/GPU 양쪽에서 실행할 수 있는 bytecode interpreter를 통해 빠른 반복과 높은 성능을 동시에 달성합니다. GPU particle 지원도 완전히 별개의 시스템을 새로 만드는 대신 기존 expression 아키텍처에 비교적 작은 변경을 가하는 방식으로 추가됐습니다. citeturn23view5turn22search2

Motion primitive는 sphere, point, plane 같은 shape를 이용해 입자의 움직임에 영향을 주는 기능입니다. 이는 단순 `position += velocity * dt`형 emitter보다 복잡한 소용돌이, 표면 회피, 폭발파, 끌림, 공간 마법 효과를 아티스트가 node 기반으로 구성할 수 있게 합니다. citeturn23view5

TFX와 particle architecture가 모두 expression, bytecode, CPU/GPU 실행, HLSL 변환이라는 유사한 개념을 사용하는 것은 우연이 아닙니다. Bungie의 렌더링 도구는 재질, particle, 동적 GPU 상태를 각각 완전히 다른 언어로 만들기보다, **아티스트가 조작하는 expression을 import 시 분석하고 런타임 실행 위치를 선택하는 공통 철학**을 갖고 있습니다. 이는 두 발표를 결합한 해석입니다. citeturn22search0turn23view5

**대기 렌더링**에서도 물리 모델과 아트 제어가 혼합됩니다. Bungie는 처음에 Bruneton·Neyret 계열의 단일·다중 산란 GPU precomputation, 우주에서의 시점, 태양·달과 light shaft를 포함하는 물리 기반 모델을 연구했지만, 이전 세대 콘솔 비용, alpha-tested geometry가 포함된 shadow volume의 확장성, 아티스트 제어의 어려움 때문에 그대로 채택하지 않았습니다. citeturn8view4

출시된 D1 접근은 대기를 두 개의 평평한 medium layer로 근사하고, optical depth를 해석적으로 계산하며, 아티스트 texture로 optical depth를 수정하는 방식이었습니다. 파장별 감쇠를 조절하고, 시간대별 sky color는 아티스트가 제작하며, 그 sky color에서 in-scattering을 근사하고 screen-space light shaft를 더했습니다. 핵심은 물리적으로 영감을 받은 산란 기반을 유지하면서도 여러 플랫폼에서 확장 가능하고 아티스트가 직접 연출할 수 있게 한 것입니다. citeturn8view4

Destiny 2와 이후 버전은 이 시스템을 발전시켰을 가능성이 높지만, 이번에 확보된 공개 자료만으로 현재의 volumetric fog grid, temporal accumulation, cloud rendering 또는 정확한 atmosphere LUT 구성을 확정할 수는 없습니다. 2020년 Bungie는 EDZ와 Nessus 일부를 당시의 최신 조명 기준에 맞게 다시 lighting하고 global-lighting update를 적용했다고 밝혔으므로, 장기 서비스 과정에서 조명·하늘 콘텐츠와 관련 파이프라인이 계속 갱신된 것은 확실합니다. citeturn16view0

**후처리와 HDR**에 대해서는 Destiny 2를 HDR display로 이식한 별도 GDC 발표가 존재하지만, 검색 가능한 공식 개요만으로는 tone curve, paper-white 기준, UI 합성, gamut mapping, SDR 역호환의 구체 수식을 확인하기 어렵습니다. 따라서 current Destiny 2가 어떤 bloom, tone mapping, temporal AA 순서를 쓰는지까지 공개 자료 이상으로 단정해서는 안 됩니다. 확정 가능한 것은 PBR·IBL 결과를 게임플레이 가독성에 맞게 art-direct하고, 별도의 HDR 출력 문제를 다룬 제작 파이프라인이 존재한다는 정도입니다. citeturn16view1turn24search8

후처리의 가장 안전한 상위 수준 복원은 다음과 같습니다.

```text
불투명 조명 결과
+ transparent / particle / additive 결과
+ atmosphere와 light shafts
→ 노출 및 장면 색 조정
→ bloom·glare 계열 효과
→ anti-aliasing 및 reconstruction
→ gameplay visibility용 color grading
→ SDR 또는 HDR display transform
→ UI와 최종 출력
```

각 블록의 정확한 순서와 중간 buffer는 플랫폼 및 시대에 따라 달라질 수 있으며, 공개 발표가 없는 부분을 Quicktag의 현재 bloom·FXAA·SSAO 구성으로 역투영해서는 안 됩니다.

## Marathon과 Quicktag에 대한 구현 명세

Destiny 2 자료를 Quicktag에 적용할 때 가장 먼저 바꿔야 할 관점은 “texture를 찾고 하나의 forward material에 넣는다”에서 “Technique가 요구하는 stage와 surface resolve를 재구축한다”로의 전환입니다. Quicktag의 현재 통합 WGSL renderer는 자산을 빠르게 확인하는 데 유용하지만, 실제 Tiger의 핵심인 G-buffer, material family, TFX runtime expression, decal stage, IBL, transparent pipeline을 하나의 fragment shader에 흡수하거나 근사합니다.

실제 Tiger에 가까워지기 위한 목표 파이프라인은 다음과 같습니다.

```text
CPU / import-side reconstruction

Tiger tag graph와 resource scope 해석
→ geometry, part, LOD, render-stage ABI 복원
→ Technique와 render state 완전 복원
→ TFX bytecode·expression·extern·object channel 복원
→ shader metadata로 vertex stream과 pass 요구조건 판정
→ Gear plate와 texture packing 복원
→ material family별 decoder 선택
→ view / frame / object / gear scope별 runtime constant 평가
```

```text
GPU frame

visibility 및 LOD 선택
→ 선택적 depth prepass
→ shadow stage
→ opaque G-buffer stage
→ alpha-tested G-buffer stage
→ ordinary decal stage
→ investment decal stage
→ deferred direct lighting
→ IBL / local environment lighting
→ additive decals
→ forward special surfaces
→ transparent geometry
→ particles와 distortion
→ atmosphere / fog / light shafts
→ exposure / post / HDR present
```

이 구조가 중요한 이유는 Gear Dye, Wear, Pattern, Decal이 모두 같은 종류의 “색상 필터”가 아니기 때문입니다. Gear Dye와 일반 wear는 대개 geometry pass에서 최종 surface property를 resolve하는 기능이고, investment decal은 별도의 geometry 또는 decal stage일 수 있으며, additive decal은 lighting 이후 합성이 적합합니다. 에너지 shield나 hologram은 아예 transparent forward stage에 속할 수 있습니다. 모든 기능을 base albedo를 차례로 수정하는 하나의 함수 체인으로 만들면 pass semantics와 blend·depth 동작이 훼손됩니다. citeturn23view2turn16view1

**G-buffer부터 먼저 도입하는 것이 우선**입니다. 정확한 bit layout을 모르는 단계에서는 고정밀 연구용 buffer를 사용해도 됩니다.

```text
RT0: linear albedo + material flags
RT1: encoded normal + roughness
RT2: metalness + AO + emissive mask + auxiliary
RT3: emissive 또는 material-family data
Depth: scene depth
```

그 뒤 실제 Tiger shader의 출력을 캡처하거나 더 많은 Technique를 분석하면서 채널을 압축하면 됩니다. 처음부터 Quicktag 고유의 압축 포맷을 추측하기보다, 각 material resolver가 생성하는 논리적 값을 검증할 수 있는 연구용 G-buffer를 만드는 편이 안전합니다.

**Gear renderer는 별도의 명시적 모듈**이어야 합니다.

```text
GearGeometryResolver
GearPlateAssembler
GearstackDecoder
GearDyeResolver
GearWearResolver
GearDetailResolver
GearDecalResolver
GearSurfaceWriter
```

`GearstackDecoder`는 RGBA를 단순히 네 값으로 반환해서는 안 되고, material family와 shader metadata에 따라 encoded alpha·blue 범위를 해석해야 합니다. `GearDyeResolver`는 default/custom/locked 그룹과 change-color index, primary/secondary branch를 모두 고려해야 합니다. `GearWearResolver`는 clean과 worn albedo뿐 아니라 roughness와 material parameters도 함께 전환해야 합니다. citeturn6view2turn23view2turn23view4

**TFX는 장기적으로 실제 interpreter가 필요합니다.** 모든 알려진 shader family를 Rust 코드의 if문으로 식별하는 접근은 초기 조사에는 빠르지만, 새 Technique가 등장할 때마다 hash와 binding 패턴을 추가해야 합니다. TFX의 stack operation, extern, object channel, scope, constant load, texture transform, output register 의미를 복원하면 Gear뿐 아니라 애니메이션 재질, weapon condition, decal selector, particle parameter와 Marathon 고유 shader도 동일한 기반에서 해석할 수 있습니다. Bungie가 TFX를 reusable component와 expression·scope 중심으로 설계한 이유도 이 확장성 때문입니다. citeturn22search0turn3view1

다만 TFX interpreter만으로 원본 렌더러가 자동 완성되지는 않습니다. TFX는 GPU 상태와 shader input을 구성하는 언어이며, 실제 compiled shader 내부의 BRDF·texture decoding·procedural pattern 수식은 별도로 복원해야 할 수 있습니다. 따라서 Quicktag에는 다음 두 층을 분리하는 것이 좋습니다.

```text
TFX Runtime
- expression과 bytecode 평가
- scope / extern / object channel 처리
- texture·sampler·constant binding
- state와 permutation 선택

Shader Family Implementation
- Gear surface
- standard world surface
- hair / cloth / skin
- investment decal
- transparent energy
- weapon wear
- Marathon-specific procedural material
```

**Animated Texture와 Shader도 같은 원칙으로 다시 접근해야 합니다.** 이전 `AnimatedDitherMaterial` 시도처럼 texture binding과 시간 변수를 추측하여 범용 WGSL에 기능을 넣기보다, 먼저 해당 Technique의 TFX expression이 어떤 extern을 읽고, 어떤 output register와 texture transform을 만들며, 어느 render stage와 blend state에서 실행되는지 복원해야 합니다. 애니메이션이 albedo resolve인지, alpha-test dither인지, emissive인지, distortion인지에 따라 패스 위치가 달라지기 때문입니다. TFX는 애초에 런타임 state와 GPU state를 expression으로 연결하도록 설계돼 있습니다. citeturn22search0turn3view1

**정확도 검증은 디버그 뷰 중심으로 구축해야 합니다.** Final color 하나만 게임 스크린샷과 비교하면 조명·노출·재질 오류가 서로 상쇄될 수 있습니다. 최소한 다음 논리 출력을 독립적으로 볼 수 있어야 합니다.

```text
part / LOD / render stage
base grayscale diffuse
dye slot과 primary-secondary 선택
resolved dye color
dye mask
wear mask
clean / worn material
detail diffuse
base / detail / final normal
smoothness / roughness
metalness
AO
alpha-test / transparency
emissive
decal contribution
G-buffer targets
direct diffuse / direct specular
IBL diffuse / IBL specular
shadow
final HDR와 display output
```

이는 첨부 이미지의 Dye, Worn Dye, AO, Roughness, Emission, Transparency, Metalness 분해와도 일치하고, Bungie가 실제 제작 과정에서 texture·mip·overdraw·pixel cost·lighting을 별도로 시각화했다는 기록과도 부합합니다. citeturn4view4turn16view1

Marathon에서는 Destiny 2의 의미를 그대로 하드코딩하기보다 **공통 Tiger 기반과 프로젝트별 확장을 분리**해야 합니다.

```text
tiger_core/
    tag graph
    buffers
    technique
    tfx
    scopes
    render states
    texture formats
    common pass graph

destiny/
    destiny gearstack
    destiny dyes
    destiny investment decals
    destiny material families

goliath/
    marathon geometry layouts
    marathon gear entities
    marathon texture passes
    marathon procedural materials
    marathon-specific TFX externs
```

이 구조는 Marathon이 Destiny와 Tiger 도구 투자를 공유한다는 공개 근거를 수용하면서도, 두 게임의 renderer가 완전히 동일하다고 가정하는 오류를 피합니다. citeturn14search3turn14search12

현재 공개 자료로 확정할 수 없는 가장 큰 항목은 최신 Destiny 2의 정확한 G-buffer 포맷, shadow filtering, local-probe blending, anti-aliasing, HDR tone mapping, redacted render stage의 의미, 그리고 Marathon 분기에서 교체된 material ABI입니다. 반면 확실하게 복원 가능한 핵심은 Tiger의 job 기반 프레임 구성, Umbra 계열 가시성, stage·LOD 기반 part 선택, deferred G-buffer 골격, PBR·IBL로의 전환, TFX의 컴포넌트·expression·scope 구조, Gear plating과 gearstack, primary·secondary·worn dye material, 독립 decal stage, 그리고 expression 기반 CPU/GPU particle architecture입니다. citeturn18view3turn23view0turn16view1turn22search0turn23view2turn23view5

종합하면 Tiger의 실제 렌더링 방식은 다음과 같이 정의할 수 있습니다.

> **Tiger는 태그 기반 콘텐츠와 자동 가시성·LOD 시스템이 선택한 geometry part를, TFX가 구성한 셰이더·GPU state·runtime expression에 따라 여러 render stage로 분배하고, 불투명 표면을 압축 G-buffer에 기록한 뒤 deferred direct lighting과 image-based lighting을 적용하며, Gear Dye·Wear·Detail·Decal을 surface resolve 및 별도 decal stage에서 합성하고, 이후 forward transparent·particle·atmosphere·display 파이프라인을 결합하는 멀티스레드 하이브리드 렌더러입니다.**

Quicktag가 실제 Tiger에 가까워지는 길은 더 많은 시각적 효과를 하나의 범용 WGSL에 추가하는 것이 아니라, **Technique–TFX–render stage–material family–G-buffer–decal–lighting의 경계를 원본 엔진처럼 다시 세우는 것**입니다.