       ŁK"	  Ŕ§ŻÖAbrain.Event:2¨-­R¤l     ąYt	łű§ŻÖA"Ů
P
PlaceholderPlaceholder*
shape:*
dtype0
*
_output_shapes
:
x
xPlaceholder*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
shape:˙˙˙˙˙˙˙˙˙
x
yPlaceholder*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
shape:˙˙˙˙˙˙˙˙˙
o
truncated_normal/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
˘
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:@*
seed2 

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:@
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:@

Variable
VariableV2*
dtype0*&
_output_shapes
:@*
	container *
shape:@*
shared_name 
Ź
Variable/AssignAssignVariabletruncated_normal*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:@
R
ConstConst*
valueB@*ÍĚĚ=*
dtype0*
_output_shapes
:@
v

Variable_1
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 

Variable_1/AssignAssign
Variable_1Const*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:@
Ě
Conv2DConv2DxVariable/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
	dilations

˛
:batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*
valueB:@*,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
:
Ł
0batch_normalization/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
: 

*batch_normalization/gamma/Initializer/onesFill:batch_normalization/gamma/Initializer/ones/shape_as_tensor0batch_normalization/gamma/Initializer/ones/Const*
T0*

index_type0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
ł
batch_normalization/gamma
VariableV2*,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
í
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(

batch_normalization/gamma/readIdentitybatch_normalization/gamma*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
ą
:batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*
valueB:@*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
˘
0batch_normalization/beta/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
: 

*batch_normalization/beta/Initializer/zerosFill:batch_normalization/beta/Initializer/zeros/shape_as_tensor0batch_normalization/beta/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ą
batch_normalization/beta
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container 
ę
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@

batch_normalization/beta/readIdentitybatch_normalization/beta*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ż
Abatch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:@*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
°
7batch_normalization/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *2
_class(
&$loc:@batch_normalization/moving_mean
 
1batch_normalization/moving_mean/Initializer/zerosFillAbatch_normalization/moving_mean/Initializer/zeros/shape_as_tensor7batch_normalization/moving_mean/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*2
_class(
&$loc:@batch_normalization/moving_mean
ż
batch_normalization/moving_mean
VariableV2*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@

&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
Ş
$batch_normalization/moving_mean/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
Ć
Dbatch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:@*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
:
ˇ
:batch_normalization/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*6
_class,
*(loc:@batch_normalization/moving_variance
­
4batch_normalization/moving_variance/Initializer/onesFillDbatch_normalization/moving_variance/Initializer/ones/shape_as_tensor:batch_normalization/moving_variance/Initializer/ones/Const*

index_type0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
T0
Ç
#batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:@

*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
validate_shape(*
_output_shapes
:@
ś
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
j
batch_normalization/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

s
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
T0
*
_output_shapes
:
q
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
_output_shapes
:*
T0

\
 batch_normalization/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
Ľ
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:12batch_normalization/cond/FusedBatchNorm/Switch_2:1batch_normalization/cond/Const batch_normalization/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training(
Ö
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchConv2D batch_normalization/cond/pred_id*
T0*
_class
loc:@Conv2D*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
Ő
0batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0*,
_class"
 loc:@batch_normalization/gamma
Ó
0batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
:@:@
Í
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_22batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( *
epsilon%o:
Ř
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchConv2D batch_normalization/cond/pred_id*
T0*
_class
loc:@Conv2D*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
×
2batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
:@:@
Ő
2batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0*+
_class!
loc:@batch_normalization/beta
ă
2batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read batch_normalization/cond/pred_id*
T0*2
_class(
&$loc:@batch_normalization/moving_mean* 
_output_shapes
:@:@
ë
2batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
Â
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: *
T0
ą
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
N*
_output_shapes

:@: *
T0
ą
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:@: 
l
!batch_normalization/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization/cond_1/switch_tIdentity#batch_normalization/cond_1/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization/cond_1/switch_fIdentity!batch_normalization/cond_1/Switch*
T0
*
_output_shapes
:
^
"batch_normalization/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization/cond_1/ConstConst$^batch_normalization/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

"batch_normalization/cond_1/Const_1Const$^batch_normalization/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 

 batch_normalization/cond_1/MergeMerge"batch_normalization/cond_1/Const_1 batch_normalization/cond_1/Const*
T0*
N*
_output_shapes
: : 
Ž
(batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
Ó
'batch_normalization/AssignMovingAvg/SubSub(batch_normalization/AssignMovingAvg/read batch_normalization/cond/Merge_1*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
Ň
'batch_normalization/AssignMovingAvg/MulMul'batch_normalization/AssignMovingAvg/Sub batch_normalization/cond_1/Merge*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
ć
#batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/Mul*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@*
use_locking( *
T0
¸
*batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
Ű
)batch_normalization/AssignMovingAvg_1/SubSub*batch_normalization/AssignMovingAvg_1/read batch_normalization/cond/Merge_2*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
Ú
)batch_normalization/AssignMovingAvg_1/MulMul)batch_normalization/AssignMovingAvg_1/Sub batch_normalization/cond_1/Merge*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
ň
%batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/Mul*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
use_locking( *
T0

BiasAddBiasAddbatch_normalization/cond/MergeVariable_1/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
q
truncated_normal_1/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=
§
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
dtype0*'
_output_shapes
:@*
seed2 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:@
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:@


Variable_2
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
ľ
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_2
x
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*'
_output_shapes
:@*
T0
V
Const_1Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_3
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

Variable_3/AssignAssign
Variable_3Const_1*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:*
use_locking(
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes	
:*
T0
×
Conv2D_1Conv2DBiasAddVariable_2/read*
paddingSAME*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ˇ
<batch_normalization_1/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_1/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_1/gamma/Initializer/onesFill<batch_normalization_1/gamma/Initializer/ones/shape_as_tensor2batch_normalization_1/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma
š
batch_normalization_1/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:
ö
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma

 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:
ś
<batch_normalization_1/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
Ś
2batch_normalization_1/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 

,batch_normalization_1/beta/Initializer/zerosFill<batch_normalization_1/beta/Initializer/zeros/shape_as_tensor2batch_normalization_1/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
ˇ
batch_normalization_1/beta
VariableV2*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ó
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes	
:*
use_locking(

batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
Ä
Cbatch_normalization_1/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_1/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_1/moving_mean/Initializer/zerosFillCbatch_normalization_1/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_1/moving_mean/Initializer/zeros/Const*

index_type0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:*
T0
Ĺ
!batch_normalization_1/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:

(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ą
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_1/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_1/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
ś
6batch_normalization_1/moving_variance/Initializer/onesFillFbatch_normalization_1/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_1/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
Í
%batch_normalization_1/moving_variance
VariableV2*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:*
dtype0

,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
l
!batch_normalization_1/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_1/cond/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


 batch_normalization_1/cond/ConstConst$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_1/cond/Const_1Const$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
ś
)batch_normalization_1/cond/FusedBatchNormFusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm/Switch:14batch_normalization_1/cond/FusedBatchNorm/Switch_1:14batch_normalization_1/cond/FusedBatchNorm/Switch_2:1 batch_normalization_1/cond/Const"batch_normalization_1/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ::::*
is_training(
ŕ
0batch_normalization_1/cond/FusedBatchNorm/SwitchSwitchConv2D_1"batch_normalization_1/cond/pred_id*
T0*
_class
loc:@Conv2D_1*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ
ß
2batch_normalization_1/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*"
_output_shapes
::
Ý
2batch_normalization_1/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*"
_output_shapes
::
Ţ
+batch_normalization_1/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm_1/Switch4batch_normalization_1/cond/FusedBatchNorm_1/Switch_14batch_normalization_1/cond/FusedBatchNorm_1/Switch_24batch_normalization_1/cond/FusedBatchNorm_1/Switch_34batch_normalization_1/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ::::*
is_training( *
epsilon%o:
â
2batch_normalization_1/cond/FusedBatchNorm_1/SwitchSwitchConv2D_1"batch_normalization_1/cond/pred_id*
T0*
_class
loc:@Conv2D_1*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ
á
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*"
_output_shapes
::
ß
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_1/beta
í
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read"batch_normalization_1/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*"
_output_shapes
::
ő
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*"
_output_shapes
::
É
 batch_normalization_1/cond/MergeMerge+batch_normalization_1/cond/FusedBatchNorm_1)batch_normalization_1/cond/FusedBatchNorm*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙ČČ: *
T0*
N
¸
"batch_normalization_1/cond/Merge_1Merge-batch_normalization_1/cond/FusedBatchNorm_1:1+batch_normalization_1/cond/FusedBatchNorm:1*
N*
_output_shapes
	:: *
T0
¸
"batch_normalization_1/cond/Merge_2Merge-batch_normalization_1/cond/FusedBatchNorm_1:2+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_1/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_1/cond_1/switch_tIdentity%batch_normalization_1/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_1/cond_1/switch_fIdentity#batch_normalization_1/cond_1/Switch*
_output_shapes
:*
T0

`
$batch_normalization_1/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_1/cond_1/ConstConst&^batch_normalization_1/cond_1/switch_t*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0

$batch_normalization_1/cond_1/Const_1Const&^batch_normalization_1/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_1/cond_1/MergeMerge$batch_normalization_1/cond_1/Const_1"batch_normalization_1/cond_1/Const*
N*
_output_shapes
: : *
T0
ľ
*batch_normalization_1/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:*
T0
Ü
)batch_normalization_1/AssignMovingAvg/SubSub*batch_normalization_1/AssignMovingAvg/read"batch_normalization_1/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_1/AssignMovingAvg/MulMul)batch_normalization_1/AssignMovingAvg/Sub"batch_normalization_1/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
ď
%batch_normalization_1/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/Mul*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:*
use_locking( 
ż
,batch_normalization_1/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
ä
+batch_normalization_1/AssignMovingAvg_1/SubSub,batch_normalization_1/AssignMovingAvg_1/read"batch_normalization_1/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
ă
+batch_normalization_1/AssignMovingAvg_1/MulMul+batch_normalization_1/AssignMovingAvg_1/Sub"batch_normalization_1/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
ű
'batch_normalization_1/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:

	BiasAdd_1BiasAdd batch_normalization_1/cond/MergeVariable_3/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ
q
truncated_normal_2/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
\
truncated_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_2/stddevConst*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0
¨
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*(
_output_shapes
:*
seed2 *

seed 

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*(
_output_shapes
:*
T0
}
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*(
_output_shapes
:


Variable_4
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ś
Variable_4/AssignAssign
Variable_4truncated_normal_2*(
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
y
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*(
_output_shapes
:
V
Const_2Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_5
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
l
Variable_5/readIdentity
Variable_5*
_output_shapes	
:*
T0*
_class
loc:@Variable_5
×
Conv2D_2Conv2D	BiasAdd_1Variable_4/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
ˇ
<batch_normalization_2/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_2/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_2/gamma/Initializer/onesFill<batch_normalization_2/gamma/Initializer/ones/shape_as_tensor2batch_normalization_2/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:
š
batch_normalization_2/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
ö
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(

 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_2/gamma
ś
<batch_normalization_2/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_2/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 

,batch_normalization_2/beta/Initializer/zerosFill<batch_normalization_2/beta/Initializer/zeros/shape_as_tensor2batch_normalization_2/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
ˇ
batch_normalization_2/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container 
ó
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
Ä
Cbatch_normalization_2/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_2/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_2/moving_mean/Initializer/zerosFillCbatch_normalization_2/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_2/moving_mean/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_2/moving_mean
Ĺ
!batch_normalization_2/moving_mean
VariableV2*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:*
dtype0

(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ą
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
Ë
Fbatch_normalization_2/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_2/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
ś
6batch_normalization_2/moving_variance/Initializer/onesFillFbatch_normalization_2/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_2/moving_variance/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_2/moving_variance
Í
%batch_normalization_2/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:

,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
l
!batch_normalization_2/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_2/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
_output_shapes
: *
valueB *
dtype0

"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:14batch_normalization_2/cond/FusedBatchNorm/Switch_2:1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training(
Ü
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchConv2D_2"batch_normalization_2/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd*
T0*
_class
loc:@Conv2D_2
ß
2batch_normalization_2/cond/FusedBatchNorm/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*"
_output_shapes
::
Ý
2batch_normalization_2/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*"
_output_shapes
::
Ü
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_24batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( 
Ţ
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchConv2D_2"batch_normalization_2/cond/pred_id*
T0*
_class
loc:@Conv2D_2*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
á
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*"
_output_shapes
::
ß
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_2/beta
í
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*"
_output_shapes
::
ő
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*"
_output_shapes
::
Ç
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: *
T0
¸
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
N*
_output_shapes
	:: *
T0
¸
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_2/cond_1/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

{
%batch_normalization_2/cond_1/switch_tIdentity%batch_normalization_2/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_2/cond_1/switch_fIdentity#batch_normalization_2/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_2/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_2/cond_1/ConstConst&^batch_normalization_2/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_2/cond_1/Const_1Const&^batch_normalization_2/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_2/cond_1/MergeMerge$batch_normalization_2/cond_1/Const_1"batch_normalization_2/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_2/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_2/AssignMovingAvg/SubSub*batch_normalization_2/AssignMovingAvg/read"batch_normalization_2/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_2/AssignMovingAvg/MulMul)batch_normalization_2/AssignMovingAvg/Sub"batch_normalization_2/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
ď
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
ż
,batch_normalization_2/AssignMovingAvg_1/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
ä
+batch_normalization_2/AssignMovingAvg_1/SubSub,batch_normalization_2/AssignMovingAvg_1/read"batch_normalization_2/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
ă
+batch_normalization_2/AssignMovingAvg_1/MulMul+batch_normalization_2/AssignMovingAvg_1/Sub"batch_normalization_2/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
ű
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:

	BiasAdd_2BiasAdd batch_normalization_2/cond/MergeVariable_5/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
q
truncated_normal_3/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_3/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*(
_output_shapes
:*
seed2 *

seed *
T0

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*(
_output_shapes
:
}
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*(
_output_shapes
:


Variable_6
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ś
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*(
_output_shapes
:
y
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*(
_output_shapes
:
V
Const_3Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_7
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes	
:*
	container 

Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
l
Variable_7/readIdentity
Variable_7*
_output_shapes	
:*
T0*
_class
loc:@Variable_7
×
Conv2D_3Conv2D	BiasAdd_2Variable_6/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
ˇ
<batch_normalization_3/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_3/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_3/gamma/Initializer/onesFill<batch_normalization_3/gamma/Initializer/ones/shape_as_tensor2batch_normalization_3/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
š
batch_normalization_3/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container 
ö
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma

 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
ś
<batch_normalization_3/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_3/beta
Ś
2batch_normalization_3/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 

,batch_normalization_3/beta/Initializer/zerosFill<batch_normalization_3/beta/Initializer/zeros/shape_as_tensor2batch_normalization_3/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
ˇ
batch_normalization_3/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ó
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_3/beta
Ä
Cbatch_normalization_3/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_3/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *4
_class*
(&loc:@batch_normalization_3/moving_mean
Š
3batch_normalization_3/moving_mean/Initializer/zerosFillCbatch_normalization_3/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_3/moving_mean/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_3/moving_mean
Ĺ
!batch_normalization_3/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container 

(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_3/moving_variance/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*
valueB:*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
ť
<batch_normalization_3/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_3/moving_variance/Initializer/onesFillFbatch_normalization_3/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_3/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:
Í
%batch_normalization_3/moving_variance
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_3/moving_variance

,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
l
!batch_normalization_3/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
_output_shapes
:*
T0

^
"batch_normalization_3/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
´
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:14batch_normalization_3/cond/FusedBatchNorm/Switch_2:1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(
Ü
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchConv2D_3"batch_normalization_3/cond/pred_id*
T0*
_class
loc:@Conv2D_3*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
ß
2batch_normalization_3/cond/FusedBatchNorm/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*"
_output_shapes
::
Ý
2batch_normalization_3/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*"
_output_shapes
::
Ü
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_24batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:*
T0
Ţ
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchConv2D_3"batch_normalization_3/cond/pred_id*
T0*
_class
loc:@Conv2D_3*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
á
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*"
_output_shapes
::
ß
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*"
_output_shapes
::
í
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_3/moving_mean/read"batch_normalization_3/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*"
_output_shapes
::
ő
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_3/moving_variance/read"batch_normalization_3/cond/pred_id*"
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
Ç
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 
¸
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
N*
_output_shapes
	:: *
T0
n
#batch_normalization_3/cond_1/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

{
%batch_normalization_3/cond_1/switch_tIdentity%batch_normalization_3/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_3/cond_1/switch_fIdentity#batch_normalization_3/cond_1/Switch*
_output_shapes
:*
T0

`
$batch_normalization_3/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_3/cond_1/ConstConst&^batch_normalization_3/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_3/cond_1/Const_1Const&^batch_normalization_3/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_3/cond_1/MergeMerge$batch_normalization_3/cond_1/Const_1"batch_normalization_3/cond_1/Const*
N*
_output_shapes
: : *
T0
ľ
*batch_normalization_3/AssignMovingAvg/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_3/AssignMovingAvg/SubSub*batch_normalization_3/AssignMovingAvg/read"batch_normalization_3/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_3/AssignMovingAvg/MulMul)batch_normalization_3/AssignMovingAvg/Sub"batch_normalization_3/cond_1/Merge*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
ď
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
ż
,batch_normalization_3/AssignMovingAvg_1/readIdentity%batch_normalization_3/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:
ä
+batch_normalization_3/AssignMovingAvg_1/SubSub,batch_normalization_3/AssignMovingAvg_1/read"batch_normalization_3/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:
ă
+batch_normalization_3/AssignMovingAvg_1/MulMul+batch_normalization_3/AssignMovingAvg_1/Sub"batch_normalization_3/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:
ű
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:

	BiasAdd_3BiasAdd batch_normalization_3/cond/MergeVariable_7/read*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0
q
truncated_normal_4/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_4/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=
¨
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*(
_output_shapes
:*
seed2 *

seed *
T0*
dtype0

truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*(
_output_shapes
:
}
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*(
_output_shapes
:*
T0


Variable_8
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ś
Variable_8/AssignAssign
Variable_8truncated_normal_4*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:
y
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*(
_output_shapes
:
V
Const_4Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_9
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

Variable_9/AssignAssign
Variable_9Const_4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_9
l
Variable_9/readIdentity
Variable_9*
_output_shapes	
:*
T0*
_class
loc:@Variable_9
×
Conv2D_4Conv2D	BiasAdd_3Variable_8/read*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ˇ
<batch_normalization_4/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_4/gamma/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0

,batch_normalization_4/gamma/Initializer/onesFill<batch_normalization_4/gamma/Initializer/ones/shape_as_tensor2batch_normalization_4/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
š
batch_normalization_4/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ö
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:

 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
ś
<batch_normalization_4/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_4/beta
Ś
2batch_normalization_4/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 

,batch_normalization_4/beta/Initializer/zerosFill<batch_normalization_4/beta/Initializer/zeros/shape_as_tensor2batch_normalization_4/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
ˇ
batch_normalization_4/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_4/beta*
	container 
ó
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
Ä
Cbatch_normalization_4/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
´
9batch_normalization_4/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_4/moving_mean/Initializer/zerosFillCbatch_normalization_4/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_4/moving_mean/Initializer/zeros/Const*

index_type0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:*
T0
Ĺ
!batch_normalization_4/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_4/moving_mean

(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:*
T0
Ë
Fbatch_normalization_4/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_4/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_4/moving_variance/Initializer/onesFillFbatch_normalization_4/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_4/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
Í
%batch_normalization_4/moving_variance
VariableV2*8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
l
!batch_normalization_4/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_4/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:14batch_normalization_4/cond/FusedBatchNorm/Switch_2:1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(
Ü
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchConv2D_4"batch_normalization_4/cond/pred_id*
_class
loc:@Conv2D_4*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0
ß
2batch_normalization_4/cond/FusedBatchNorm/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*"
_output_shapes
::
Ý
2batch_normalization_4/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_4/beta
Ü
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_24batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:*
T0*
data_formatNHWC
Ţ
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchConv2D_4"batch_normalization_4/cond/pred_id*
T0*
_class
loc:@Conv2D_4*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
á
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_4/gamma
ß
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_4/beta
í
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_4/moving_mean/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
ő
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_4/moving_variance/read"batch_normalization_4/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*"
_output_shapes
::
Ç
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 
¸
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
N*
_output_shapes
	:: *
T0
¸
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
N*
_output_shapes
	:: *
T0
n
#batch_normalization_4/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_4/cond_1/switch_tIdentity%batch_normalization_4/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_4/cond_1/switch_fIdentity#batch_normalization_4/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_4/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_4/cond_1/ConstConst&^batch_normalization_4/cond_1/switch_t*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0

$batch_normalization_4/cond_1/Const_1Const&^batch_normalization_4/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Ą
"batch_normalization_4/cond_1/MergeMerge$batch_normalization_4/cond_1/Const_1"batch_normalization_4/cond_1/Const*
_output_shapes
: : *
T0*
N
ľ
*batch_normalization_4/AssignMovingAvg/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_4/AssignMovingAvg/SubSub*batch_normalization_4/AssignMovingAvg/read"batch_normalization_4/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_4/AssignMovingAvg/MulMul)batch_normalization_4/AssignMovingAvg/Sub"batch_normalization_4/cond_1/Merge*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
ď
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:
ż
,batch_normalization_4/AssignMovingAvg_1/readIdentity%batch_normalization_4/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
ä
+batch_normalization_4/AssignMovingAvg_1/SubSub,batch_normalization_4/AssignMovingAvg_1/read"batch_normalization_4/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
ă
+batch_normalization_4/AssignMovingAvg_1/MulMul+batch_normalization_4/AssignMovingAvg_1/Sub"batch_normalization_4/cond_1/Merge*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
ű
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:

	BiasAdd_4BiasAdd batch_normalization_4/cond/MergeVariable_9/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
q
truncated_normal_5/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_5/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*(
_output_shapes
:*
seed2 *

seed *
T0

truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*(
_output_shapes
:
}
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*(
_output_shapes
:*
T0

Variable_10
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
š
Variable_10/AssignAssignVariable_10truncated_normal_5*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*(
_output_shapes
:
|
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*(
_output_shapes
:
V
Const_5Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
y
Variable_11
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ą
Variable_11/AssignAssignVariable_11Const_5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_11
o
Variable_11/readIdentityVariable_11*
_output_shapes	
:*
T0*
_class
loc:@Variable_11
Ř
Conv2D_5Conv2D	BiasAdd_4Variable_10/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
	dilations

ˇ
<batch_normalization_5/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_5/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_5/gamma/Initializer/onesFill<batch_normalization_5/gamma/Initializer/ones/shape_as_tensor2batch_normalization_5/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma
š
batch_normalization_5/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma*
	container *
shape:
ö
"batch_normalization_5/gamma/AssignAssignbatch_normalization_5/gamma,batch_normalization_5/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:

 batch_normalization_5/gamma/readIdentitybatch_normalization_5/gamma*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_5/gamma
ś
<batch_normalization_5/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_5/beta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta*
dtype0

,batch_normalization_5/beta/Initializer/zerosFill<batch_normalization_5/beta/Initializer/zeros/shape_as_tensor2batch_normalization_5/beta/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta
ˇ
batch_normalization_5/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_5/beta*
	container *
shape:
ó
!batch_normalization_5/beta/AssignAssignbatch_normalization_5/beta,batch_normalization_5/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

batch_normalization_5/beta/readIdentitybatch_normalization_5/beta*
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Ä
Cbatch_normalization_5/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_5/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_5/moving_mean/Initializer/zerosFillCbatch_normalization_5/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_5/moving_mean/Initializer/zeros/Const*

index_type0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:*
T0
Ĺ
!batch_normalization_5/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_5/moving_mean*
	container 

(batch_normalization_5/moving_mean/AssignAssign!batch_normalization_5/moving_mean3batch_normalization_5/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_5/moving_mean/readIdentity!batch_normalization_5/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_5/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_5/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0
ś
6batch_normalization_5/moving_variance/Initializer/onesFillFbatch_normalization_5/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_5/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:
Í
%batch_normalization_5/moving_variance
VariableV2*8
_class.
,*loc:@batch_normalization_5/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

,batch_normalization_5/moving_variance/AssignAssign%batch_normalization_5/moving_variance6batch_normalization_5/moving_variance/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
validate_shape(
˝
*batch_normalization_5/moving_variance/readIdentity%batch_normalization_5/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:
l
!batch_normalization_5/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_5/cond/switch_tIdentity#batch_normalization_5/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_5/cond/switch_fIdentity!batch_normalization_5/cond/Switch*
_output_shapes
:*
T0

^
"batch_normalization_5/cond/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


 batch_normalization_5/cond/ConstConst$^batch_normalization_5/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_5/cond/Const_1Const$^batch_normalization_5/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
´
)batch_normalization_5/cond/FusedBatchNormFusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:14batch_normalization_5/cond/FusedBatchNorm/Switch_2:1 batch_normalization_5/cond/Const"batch_normalization_5/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(
Ü
0batch_normalization_5/cond/FusedBatchNorm/SwitchSwitchConv2D_5"batch_normalization_5/cond/pred_id*
T0*
_class
loc:@Conv2D_5*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
ß
2batch_normalization_5/cond/FusedBatchNorm/Switch_1Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_5/gamma*"
_output_shapes
::
Ý
2batch_normalization_5/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_5/beta
Ü
+batch_normalization_5/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_24batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:*
T0
Ţ
2batch_normalization_5/cond/FusedBatchNorm_1/SwitchSwitchConv2D_5"batch_normalization_5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0*
_class
loc:@Conv2D_5
á
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_5/gamma*"
_output_shapes
::
ß
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_5/beta*"
_output_shapes
::
í
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_5/moving_mean/read"batch_normalization_5/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*"
_output_shapes
::
ő
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_5/moving_variance/read"batch_normalization_5/cond/pred_id*"
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
Ç
 batch_normalization_5/cond/MergeMerge+batch_normalization_5/cond/FusedBatchNorm_1)batch_normalization_5/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 
¸
"batch_normalization_5/cond/Merge_1Merge-batch_normalization_5/cond/FusedBatchNorm_1:1+batch_normalization_5/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_5/cond/Merge_2Merge-batch_normalization_5/cond/FusedBatchNorm_1:2+batch_normalization_5/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_5/cond_1/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

{
%batch_normalization_5/cond_1/switch_tIdentity%batch_normalization_5/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_5/cond_1/switch_fIdentity#batch_normalization_5/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_5/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_5/cond_1/ConstConst&^batch_normalization_5/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_5/cond_1/Const_1Const&^batch_normalization_5/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_5/cond_1/MergeMerge$batch_normalization_5/cond_1/Const_1"batch_normalization_5/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_5/AssignMovingAvg/readIdentity!batch_normalization_5/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_5/AssignMovingAvg/SubSub*batch_normalization_5/AssignMovingAvg/read"batch_normalization_5/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_5/AssignMovingAvg/MulMul)batch_normalization_5/AssignMovingAvg/Sub"batch_normalization_5/cond_1/Merge*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
ď
%batch_normalization_5/AssignMovingAvg	AssignSub!batch_normalization_5/moving_mean)batch_normalization_5/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
ż
,batch_normalization_5/AssignMovingAvg_1/readIdentity%batch_normalization_5/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:
ä
+batch_normalization_5/AssignMovingAvg_1/SubSub,batch_normalization_5/AssignMovingAvg_1/read"batch_normalization_5/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:
ă
+batch_normalization_5/AssignMovingAvg_1/MulMul+batch_normalization_5/AssignMovingAvg_1/Sub"batch_normalization_5/cond_1/Merge*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
ű
'batch_normalization_5/AssignMovingAvg_1	AssignSub%batch_normalization_5/moving_variance+batch_normalization_5/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:

	BiasAdd_5BiasAdd batch_normalization_5/cond/MergeVariable_11/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
q
truncated_normal_6/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_6/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=
¨
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*(
_output_shapes
:*
seed2 *

seed *
T0*
dtype0

truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*(
_output_shapes
:
}
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0*(
_output_shapes
:

Variable_12
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
š
Variable_12/AssignAssignVariable_12truncated_normal_6*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*(
_output_shapes
:
|
Variable_12/readIdentityVariable_12*(
_output_shapes
:*
T0*
_class
loc:@Variable_12
V
Const_6Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
y
Variable_13
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ą
Variable_13/AssignAssignVariable_13Const_6*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(
o
Variable_13/readIdentityVariable_13*
_output_shapes	
:*
T0*
_class
loc:@Variable_13
Ř
Conv2D_6Conv2D	BiasAdd_5Variable_12/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
	dilations

ˇ
<batch_normalization_6/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_6/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_6/gamma/Initializer/onesFill<batch_normalization_6/gamma/Initializer/ones/shape_as_tensor2batch_normalization_6/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_6/gamma
š
batch_normalization_6/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_6/gamma*
	container *
shape:
ö
"batch_normalization_6/gamma/AssignAssignbatch_normalization_6/gamma,batch_normalization_6/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:

 batch_normalization_6/gamma/readIdentitybatch_normalization_6/gamma*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
ś
<batch_normalization_6/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_6/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
: 

,batch_normalization_6/beta/Initializer/zerosFill<batch_normalization_6/beta/Initializer/zeros/shape_as_tensor2batch_normalization_6/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
ˇ
batch_normalization_6/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_6/beta*
	container 
ó
!batch_normalization_6/beta/AssignAssignbatch_normalization_6/beta,batch_normalization_6/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_6/beta/readIdentitybatch_normalization_6/beta*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:*
T0
Ä
Cbatch_normalization_6/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*4
_class*
(&loc:@batch_normalization_6/moving_mean*
dtype0
´
9batch_normalization_6/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_6/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_6/moving_mean/Initializer/zerosFillCbatch_normalization_6/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_6/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
Ĺ
!batch_normalization_6/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_6/moving_mean*
	container 

(batch_normalization_6/moving_mean/AssignAssign!batch_normalization_6/moving_mean3batch_normalization_6/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_6/moving_mean/readIdentity!batch_normalization_6/moving_mean*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean
Ë
Fbatch_normalization_6/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_6/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0
ś
6batch_normalization_6/moving_variance/Initializer/onesFillFbatch_normalization_6/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_6/moving_variance/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_6/moving_variance
Í
%batch_normalization_6/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_6/moving_variance*
	container 

,batch_normalization_6/moving_variance/AssignAssign%batch_normalization_6/moving_variance6batch_normalization_6/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_6/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˝
*batch_normalization_6/moving_variance/readIdentity%batch_normalization_6/moving_variance*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance
l
!batch_normalization_6/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_6/cond/switch_tIdentity#batch_normalization_6/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_6/cond/switch_fIdentity!batch_normalization_6/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_6/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_6/cond/ConstConst$^batch_normalization_6/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 

"batch_normalization_6/cond/Const_1Const$^batch_normalization_6/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_6/cond/FusedBatchNormFusedBatchNorm2batch_normalization_6/cond/FusedBatchNorm/Switch:14batch_normalization_6/cond/FusedBatchNorm/Switch_1:14batch_normalization_6/cond/FusedBatchNorm/Switch_2:1 batch_normalization_6/cond/Const"batch_normalization_6/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(
Ü
0batch_normalization_6/cond/FusedBatchNorm/SwitchSwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*
_class
loc:@Conv2D_6*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
ß
2batch_normalization_6/cond/FusedBatchNorm/Switch_1Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_6/gamma
Ý
2batch_normalization_6/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_6/beta*"
_output_shapes
::
Ü
+batch_normalization_6/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_6/cond/FusedBatchNorm_1/Switch4batch_normalization_6/cond/FusedBatchNorm_1/Switch_14batch_normalization_6/cond/FusedBatchNorm_1/Switch_24batch_normalization_6/cond/FusedBatchNorm_1/Switch_34batch_normalization_6/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( 
Ţ
2batch_normalization_6/cond/FusedBatchNorm_1/SwitchSwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*
_class
loc:@Conv2D_6*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
á
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_6/gamma
ß
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_6/beta*"
_output_shapes
::
í
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_6/moving_mean/read"batch_normalization_6/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*"
_output_shapes
::
ő
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_6/moving_variance/read"batch_normalization_6/cond/pred_id*"
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance
Ç
 batch_normalization_6/cond/MergeMerge+batch_normalization_6/cond/FusedBatchNorm_1)batch_normalization_6/cond/FusedBatchNorm*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: *
T0
¸
"batch_normalization_6/cond/Merge_1Merge-batch_normalization_6/cond/FusedBatchNorm_1:1+batch_normalization_6/cond/FusedBatchNorm:1*
N*
_output_shapes
	:: *
T0
¸
"batch_normalization_6/cond/Merge_2Merge-batch_normalization_6/cond/FusedBatchNorm_1:2+batch_normalization_6/cond/FusedBatchNorm:2*
N*
_output_shapes
	:: *
T0
n
#batch_normalization_6/cond_1/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

{
%batch_normalization_6/cond_1/switch_tIdentity%batch_normalization_6/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_6/cond_1/switch_fIdentity#batch_normalization_6/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_6/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_6/cond_1/ConstConst&^batch_normalization_6/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=

$batch_normalization_6/cond_1/Const_1Const&^batch_normalization_6/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Ą
"batch_normalization_6/cond_1/MergeMerge$batch_normalization_6/cond_1/Const_1"batch_normalization_6/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_6/AssignMovingAvg/readIdentity!batch_normalization_6/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_6/AssignMovingAvg/SubSub*batch_normalization_6/AssignMovingAvg/read"batch_normalization_6/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_6/AssignMovingAvg/MulMul)batch_normalization_6/AssignMovingAvg/Sub"batch_normalization_6/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
ď
%batch_normalization_6/AssignMovingAvg	AssignSub!batch_normalization_6/moving_mean)batch_normalization_6/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
ż
,batch_normalization_6/AssignMovingAvg_1/readIdentity%batch_normalization_6/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:
ä
+batch_normalization_6/AssignMovingAvg_1/SubSub,batch_normalization_6/AssignMovingAvg_1/read"batch_normalization_6/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:
ă
+batch_normalization_6/AssignMovingAvg_1/MulMul+batch_normalization_6/AssignMovingAvg_1/Sub"batch_normalization_6/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:
ű
'batch_normalization_6/AssignMovingAvg_1	AssignSub%batch_normalization_6/moving_variance+batch_normalization_6/AssignMovingAvg_1/Mul*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:*
use_locking( *
T0

	BiasAdd_6BiasAdd batch_normalization_6/cond/MergeVariable_13/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
k
ResizeNearestNeighbor/sizeConst*
valueB"d   d   *
dtype0*
_output_shapes
:
Ľ
ResizeNearestNeighborResizeNearestNeighbor	BiasAdd_6ResizeNearestNeighbor/size*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
align_corners( 
q
truncated_normal_7/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_7/stddevConst*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0
¨
"truncated_normal_7/TruncatedNormalTruncatedNormaltruncated_normal_7/shape*

seed *
T0*
dtype0*(
_output_shapes
:*
seed2 

truncated_normal_7/mulMul"truncated_normal_7/TruncatedNormaltruncated_normal_7/stddev*
T0*(
_output_shapes
:
}
truncated_normal_7Addtruncated_normal_7/multruncated_normal_7/mean*
T0*(
_output_shapes
:

Variable_14
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
š
Variable_14/AssignAssignVariable_14truncated_normal_7*
T0*
_class
loc:@Variable_14*
validate_shape(*(
_output_shapes
:*
use_locking(
|
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14*(
_output_shapes
:
V
Const_7Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
y
Variable_15
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ą
Variable_15/AssignAssignVariable_15Const_7*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_15
o
Variable_15/readIdentityVariable_15*
T0*
_class
loc:@Variable_15*
_output_shapes	
:
ä
Conv2D_7Conv2DResizeNearestNeighborVariable_14/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ˇ
<batch_normalization_7/gamma/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*
valueB:*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0
§
2batch_normalization_7/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_7/gamma/Initializer/onesFill<batch_normalization_7/gamma/Initializer/ones/shape_as_tensor2batch_normalization_7/gamma/Initializer/ones/Const*

index_type0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:*
T0
š
batch_normalization_7/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_7/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ö
"batch_normalization_7/gamma/AssignAssignbatch_normalization_7/gamma,batch_normalization_7/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_7/gamma

 batch_normalization_7/gamma/readIdentitybatch_normalization_7/gamma*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
ś
<batch_normalization_7/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_7/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
: 

,batch_normalization_7/beta/Initializer/zerosFill<batch_normalization_7/beta/Initializer/zeros/shape_as_tensor2batch_normalization_7/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
ˇ
batch_normalization_7/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_7/beta*
	container 
ó
!batch_normalization_7/beta/AssignAssignbatch_normalization_7/beta,batch_normalization_7/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_7/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

batch_normalization_7/beta/readIdentitybatch_normalization_7/beta*
T0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
Ä
Cbatch_normalization_7/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_7/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_7/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_7/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_7/moving_mean/Initializer/zerosFillCbatch_normalization_7/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_7/moving_mean/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_7/moving_mean
Ĺ
!batch_normalization_7/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_7/moving_mean*
	container 

(batch_normalization_7/moving_mean/AssignAssign!batch_normalization_7/moving_mean3batch_normalization_7/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_7/moving_mean/readIdentity!batch_normalization_7/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_7/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_7/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_7/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_7/moving_variance*
dtype0
ś
6batch_normalization_7/moving_variance/Initializer/onesFillFbatch_normalization_7/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_7/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
Í
%batch_normalization_7/moving_variance
VariableV2*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_7/moving_variance*
	container *
shape:*
dtype0

,batch_normalization_7/moving_variance/AssignAssign%batch_normalization_7/moving_variance6batch_normalization_7/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_7/moving_variance/readIdentity%batch_normalization_7/moving_variance*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance
l
!batch_normalization_7/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

w
#batch_normalization_7/cond/switch_tIdentity#batch_normalization_7/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_7/cond/switch_fIdentity!batch_normalization_7/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_7/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_7/cond/ConstConst$^batch_normalization_7/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_7/cond/Const_1Const$^batch_normalization_7/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_7/cond/FusedBatchNormFusedBatchNorm2batch_normalization_7/cond/FusedBatchNorm/Switch:14batch_normalization_7/cond/FusedBatchNorm/Switch_1:14batch_normalization_7/cond/FusedBatchNorm/Switch_2:1 batch_normalization_7/cond/Const"batch_normalization_7/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training(
Ü
0batch_normalization_7/cond/FusedBatchNorm/SwitchSwitchConv2D_7"batch_normalization_7/cond/pred_id*
T0*
_class
loc:@Conv2D_7*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
ß
2batch_normalization_7/cond/FusedBatchNorm/Switch_1Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_7/gamma*"
_output_shapes
::
Ý
2batch_normalization_7/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_7/beta*"
_output_shapes
::
Ü
+batch_normalization_7/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_7/cond/FusedBatchNorm_1/Switch4batch_normalization_7/cond/FusedBatchNorm_1/Switch_14batch_normalization_7/cond/FusedBatchNorm_1/Switch_24batch_normalization_7/cond/FusedBatchNorm_1/Switch_34batch_normalization_7/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( *
epsilon%o:
Ţ
2batch_normalization_7/cond/FusedBatchNorm_1/SwitchSwitchConv2D_7"batch_normalization_7/cond/pred_id*
T0*
_class
loc:@Conv2D_7*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
á
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_7/gamma
ß
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_7/beta*"
_output_shapes
::
í
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_7/moving_mean/read"batch_normalization_7/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*"
_output_shapes
::
ő
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_7/moving_variance/read"batch_normalization_7/cond/pred_id*8
_class.
,*loc:@batch_normalization_7/moving_variance*"
_output_shapes
::*
T0
Ç
 batch_normalization_7/cond/MergeMerge+batch_normalization_7/cond/FusedBatchNorm_1)batch_normalization_7/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 
¸
"batch_normalization_7/cond/Merge_1Merge-batch_normalization_7/cond/FusedBatchNorm_1:1+batch_normalization_7/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_7/cond/Merge_2Merge-batch_normalization_7/cond/FusedBatchNorm_1:2+batch_normalization_7/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_7/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_7/cond_1/switch_tIdentity%batch_normalization_7/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_7/cond_1/switch_fIdentity#batch_normalization_7/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_7/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_7/cond_1/ConstConst&^batch_normalization_7/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=

$batch_normalization_7/cond_1/Const_1Const&^batch_normalization_7/cond_1/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ą
"batch_normalization_7/cond_1/MergeMerge$batch_normalization_7/cond_1/Const_1"batch_normalization_7/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_7/AssignMovingAvg/readIdentity!batch_normalization_7/moving_mean*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean
Ü
)batch_normalization_7/AssignMovingAvg/SubSub*batch_normalization_7/AssignMovingAvg/read"batch_normalization_7/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_7/AssignMovingAvg/MulMul)batch_normalization_7/AssignMovingAvg/Sub"batch_normalization_7/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
ď
%batch_normalization_7/AssignMovingAvg	AssignSub!batch_normalization_7/moving_mean)batch_normalization_7/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
ż
,batch_normalization_7/AssignMovingAvg_1/readIdentity%batch_normalization_7/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
ä
+batch_normalization_7/AssignMovingAvg_1/SubSub,batch_normalization_7/AssignMovingAvg_1/read"batch_normalization_7/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
ă
+batch_normalization_7/AssignMovingAvg_1/MulMul+batch_normalization_7/AssignMovingAvg_1/Sub"batch_normalization_7/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
ű
'batch_normalization_7/AssignMovingAvg_1	AssignSub%batch_normalization_7/moving_variance+batch_normalization_7/AssignMovingAvg_1/Mul*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:*
use_locking( 

	BiasAdd_7BiasAdd batch_normalization_7/cond/MergeVariable_15/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
m
ResizeNearestNeighbor_1/sizeConst*
valueB"Č   Č   *
dtype0*
_output_shapes
:
Ť
ResizeNearestNeighbor_1ResizeNearestNeighbor	BiasAdd_7ResizeNearestNeighbor_1/size*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
align_corners( 
q
truncated_normal_8/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
\
truncated_normal_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_8/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=
§
"truncated_normal_8/TruncatedNormalTruncatedNormaltruncated_normal_8/shape*
dtype0*'
_output_shapes
:@*
seed2 *

seed *
T0

truncated_normal_8/mulMul"truncated_normal_8/TruncatedNormaltruncated_normal_8/stddev*
T0*'
_output_shapes
:@
|
truncated_normal_8Addtruncated_normal_8/multruncated_normal_8/mean*
T0*'
_output_shapes
:@

Variable_16
VariableV2*'
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
¸
Variable_16/AssignAssignVariable_16truncated_normal_8*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*'
_output_shapes
:@
{
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16*'
_output_shapes
:@
T
Const_8Const*
valueB@*ÍĚĚ=*
dtype0*
_output_shapes
:@
w
Variable_17
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
 
Variable_17/AssignAssignVariable_17Const_8*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:@
n
Variable_17/readIdentityVariable_17*
_output_shapes
:@*
T0*
_class
loc:@Variable_17
ç
Conv2D_8Conv2DResizeNearestNeighbor_1Variable_16/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
	dilations

ś
<batch_normalization_8/gamma/Initializer/ones/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_8/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_8/gamma/Initializer/onesFill<batch_normalization_8/gamma/Initializer/ones/shape_as_tensor2batch_normalization_8/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
ˇ
batch_normalization_8/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *.
_class$
" loc:@batch_normalization_8/gamma*
	container *
shape:@
ő
"batch_normalization_8/gamma/AssignAssignbatch_normalization_8/gamma,batch_normalization_8/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
validate_shape(*
_output_shapes
:@

 batch_normalization_8/gamma/readIdentitybatch_normalization_8/gamma*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
ľ
<batch_normalization_8/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:@*-
_class#
!loc:@batch_normalization_8/beta*
dtype0
Ś
2batch_normalization_8/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
_output_shapes
: 

,batch_normalization_8/beta/Initializer/zerosFill<batch_normalization_8/beta/Initializer/zeros/shape_as_tensor2batch_normalization_8/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
ľ
batch_normalization_8/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_8/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
ň
!batch_normalization_8/beta/AssignAssignbatch_normalization_8/beta,batch_normalization_8/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_8/beta*
validate_shape(*
_output_shapes
:@

batch_normalization_8/beta/readIdentitybatch_normalization_8/beta*
T0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
Ă
Cbatch_normalization_8/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:@*4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_8/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0*
_output_shapes
: 
¨
3batch_normalization_8/moving_mean/Initializer/zerosFillCbatch_normalization_8/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_8/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
Ă
!batch_normalization_8/moving_mean
VariableV2*
shared_name *4
_class*
(&loc:@batch_normalization_8/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@

(batch_normalization_8/moving_mean/AssignAssign!batch_normalization_8/moving_mean3batch_normalization_8/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
validate_shape(*
_output_shapes
:@
°
&batch_normalization_8/moving_mean/readIdentity!batch_normalization_8/moving_mean*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@*
T0
Ę
Fbatch_normalization_8/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:@*8
_class.
,*loc:@batch_normalization_8/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_8/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_8/moving_variance
ľ
6batch_normalization_8/moving_variance/Initializer/onesFillFbatch_normalization_8/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_8/moving_variance/Initializer/ones/Const*
_output_shapes
:@*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_8/moving_variance
Ë
%batch_normalization_8/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_8/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@

,batch_normalization_8/moving_variance/AssignAssign%batch_normalization_8/moving_variance6batch_normalization_8/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_8/moving_variance*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ź
*batch_normalization_8/moving_variance/readIdentity%batch_normalization_8/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@
l
!batch_normalization_8/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_8/cond/switch_tIdentity#batch_normalization_8/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_8/cond/switch_fIdentity!batch_normalization_8/cond/Switch*
_output_shapes
:*
T0

^
"batch_normalization_8/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_8/cond/ConstConst$^batch_normalization_8/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_8/cond/Const_1Const$^batch_normalization_8/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
ą
)batch_normalization_8/cond/FusedBatchNormFusedBatchNorm2batch_normalization_8/cond/FusedBatchNorm/Switch:14batch_normalization_8/cond/FusedBatchNorm/Switch_1:14batch_normalization_8/cond/FusedBatchNorm/Switch_2:1 batch_normalization_8/cond/Const"batch_normalization_8/cond/Const_1*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training(*
epsilon%o:
Ţ
0batch_normalization_8/cond/FusedBatchNorm/SwitchSwitchConv2D_8"batch_normalization_8/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0*
_class
loc:@Conv2D_8
Ý
2batch_normalization_8/cond/FusedBatchNorm/Switch_1Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_8/gamma* 
_output_shapes
:@:@
Ű
2batch_normalization_8/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_8/beta* 
_output_shapes
:@:@
Ů
+batch_normalization_8/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_8/cond/FusedBatchNorm_1/Switch4batch_normalization_8/cond/FusedBatchNorm_1/Switch_14batch_normalization_8/cond/FusedBatchNorm_1/Switch_24batch_normalization_8/cond/FusedBatchNorm_1/Switch_34batch_normalization_8/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( 
ŕ
2batch_normalization_8/cond/FusedBatchNorm_1/SwitchSwitchConv2D_8"batch_normalization_8/cond/pred_id*
T0*
_class
loc:@Conv2D_8*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
ß
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_8/gamma* 
_output_shapes
:@:@
Ý
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_8/beta* 
_output_shapes
:@:@
ë
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_8/moving_mean/read"batch_normalization_8/cond/pred_id*4
_class*
(&loc:@batch_normalization_8/moving_mean* 
_output_shapes
:@:@*
T0
ó
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_8/moving_variance/read"batch_normalization_8/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance* 
_output_shapes
:@:@
Č
 batch_normalization_8/cond/MergeMerge+batch_normalization_8/cond/FusedBatchNorm_1)batch_normalization_8/cond/FusedBatchNorm*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 
ˇ
"batch_normalization_8/cond/Merge_1Merge-batch_normalization_8/cond/FusedBatchNorm_1:1+batch_normalization_8/cond/FusedBatchNorm:1*
N*
_output_shapes

:@: *
T0
ˇ
"batch_normalization_8/cond/Merge_2Merge-batch_normalization_8/cond/FusedBatchNorm_1:2+batch_normalization_8/cond/FusedBatchNorm:2*
N*
_output_shapes

:@: *
T0
n
#batch_normalization_8/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_8/cond_1/switch_tIdentity%batch_normalization_8/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_8/cond_1/switch_fIdentity#batch_normalization_8/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_8/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_8/cond_1/ConstConst&^batch_normalization_8/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_8/cond_1/Const_1Const&^batch_normalization_8/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Ą
"batch_normalization_8/cond_1/MergeMerge$batch_normalization_8/cond_1/Const_1"batch_normalization_8/cond_1/Const*
T0*
N*
_output_shapes
: : 
´
*batch_normalization_8/AssignMovingAvg/readIdentity!batch_normalization_8/moving_mean*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@*
T0
Ű
)batch_normalization_8/AssignMovingAvg/SubSub*batch_normalization_8/AssignMovingAvg/read"batch_normalization_8/cond/Merge_1*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@*
T0
Ú
)batch_normalization_8/AssignMovingAvg/MulMul)batch_normalization_8/AssignMovingAvg/Sub"batch_normalization_8/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
î
%batch_normalization_8/AssignMovingAvg	AssignSub!batch_normalization_8/moving_mean)batch_normalization_8/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
ž
,batch_normalization_8/AssignMovingAvg_1/readIdentity%batch_normalization_8/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@
ă
+batch_normalization_8/AssignMovingAvg_1/SubSub,batch_normalization_8/AssignMovingAvg_1/read"batch_normalization_8/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@
â
+batch_normalization_8/AssignMovingAvg_1/MulMul+batch_normalization_8/AssignMovingAvg_1/Sub"batch_normalization_8/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@
ú
'batch_normalization_8/AssignMovingAvg_1	AssignSub%batch_normalization_8/moving_variance+batch_normalization_8/AssignMovingAvg_1/Mul*
_output_shapes
:@*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance

	BiasAdd_8BiasAdd batch_normalization_8/cond/MergeVariable_17/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
T0*
data_formatNHWC
m
ResizeNearestNeighbor_2/sizeConst*
valueB"    *
dtype0*
_output_shapes
:
Ş
ResizeNearestNeighbor_2ResizeNearestNeighbor	BiasAdd_8ResizeNearestNeighbor_2/size*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
align_corners( *
T0
q
truncated_normal_9/shapeConst*%
valueB"      @       *
dtype0*
_output_shapes
:
\
truncated_normal_9/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
^
truncated_normal_9/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
Ś
"truncated_normal_9/TruncatedNormalTruncatedNormaltruncated_normal_9/shape*

seed *
T0*
dtype0*&
_output_shapes
:@ *
seed2 

truncated_normal_9/mulMul"truncated_normal_9/TruncatedNormaltruncated_normal_9/stddev*
T0*&
_output_shapes
:@ 
{
truncated_normal_9Addtruncated_normal_9/multruncated_normal_9/mean*
T0*&
_output_shapes
:@ 

Variable_18
VariableV2*
shape:@ *
shared_name *
dtype0*&
_output_shapes
:@ *
	container 
ˇ
Variable_18/AssignAssignVariable_18truncated_normal_9*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:@ 
z
Variable_18/readIdentityVariable_18*&
_output_shapes
:@ *
T0*
_class
loc:@Variable_18
T
Const_9Const*
dtype0*
_output_shapes
: *
valueB *ÍĚĚ=
w
Variable_19
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
 
Variable_19/AssignAssignVariable_19Const_9*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(
n
Variable_19/readIdentityVariable_19*
T0*
_class
loc:@Variable_19*
_output_shapes
: 
ç
Conv2D_9Conv2DResizeNearestNeighbor_2Variable_18/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ś
<batch_normalization_9/gamma/Initializer/ones/shape_as_tensorConst*
valueB: *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_9/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_9/gamma/Initializer/onesFill<batch_normalization_9/gamma/Initializer/ones/shape_as_tensor2batch_normalization_9/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
ˇ
batch_normalization_9/gamma
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_9/gamma*
	container *
shape: 
ő
"batch_normalization_9/gamma/AssignAssignbatch_normalization_9/gamma,batch_normalization_9/gamma/Initializer/ones*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@batch_normalization_9/gamma

 batch_normalization_9/gamma/readIdentitybatch_normalization_9/gamma*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_9/gamma
ľ
<batch_normalization_9/beta/Initializer/zeros/shape_as_tensorConst*
valueB: *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_9/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
: 

,batch_normalization_9/beta/Initializer/zerosFill<batch_normalization_9/beta/Initializer/zeros/shape_as_tensor2batch_normalization_9/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
ľ
batch_normalization_9/beta
VariableV2*-
_class#
!loc:@batch_normalization_9/beta*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
ň
!batch_normalization_9/beta/AssignAssignbatch_normalization_9/beta,batch_normalization_9/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_9/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

batch_normalization_9/beta/readIdentitybatch_normalization_9/beta*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_9/beta
Ă
Cbatch_normalization_9/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB: *4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_9/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0*
_output_shapes
: 
¨
3batch_normalization_9/moving_mean/Initializer/zerosFillCbatch_normalization_9/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_9/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ă
!batch_normalization_9/moving_mean
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@batch_normalization_9/moving_mean*
	container *
shape: 

(batch_normalization_9/moving_mean/AssignAssign!batch_normalization_9/moving_mean3batch_normalization_9/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
validate_shape(*
_output_shapes
: 
°
&batch_normalization_9/moving_mean/readIdentity!batch_normalization_9/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ę
Fbatch_normalization_9/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *8
_class.
,*loc:@batch_normalization_9/moving_variance
ť
<batch_normalization_9/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_9/moving_variance
ľ
6batch_normalization_9/moving_variance/Initializer/onesFillFbatch_normalization_9/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_9/moving_variance/Initializer/ones/Const*
_output_shapes
: *
T0*

index_type0*8
_class.
,*loc:@batch_normalization_9/moving_variance
Ë
%batch_normalization_9/moving_variance
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *8
_class.
,*loc:@batch_normalization_9/moving_variance

,batch_normalization_9/moving_variance/AssignAssign%batch_normalization_9/moving_variance6batch_normalization_9/moving_variance/Initializer/ones*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
ź
*batch_normalization_9/moving_variance/readIdentity%batch_normalization_9/moving_variance*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance
l
!batch_normalization_9/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

w
#batch_normalization_9/cond/switch_tIdentity#batch_normalization_9/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_9/cond/switch_fIdentity!batch_normalization_9/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_9/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_9/cond/ConstConst$^batch_normalization_9/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_9/cond/Const_1Const$^batch_normalization_9/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
ą
)batch_normalization_9/cond/FusedBatchNormFusedBatchNorm2batch_normalization_9/cond/FusedBatchNorm/Switch:14batch_normalization_9/cond/FusedBatchNorm/Switch_1:14batch_normalization_9/cond/FusedBatchNorm/Switch_2:1 batch_normalization_9/cond/Const"batch_normalization_9/cond/Const_1*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
is_training(*
epsilon%o:
Ţ
0batch_normalization_9/cond/FusedBatchNorm/SwitchSwitchConv2D_9"batch_normalization_9/cond/pred_id*
T0*
_class
loc:@Conv2D_9*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
Ý
2batch_normalization_9/cond/FusedBatchNorm/Switch_1Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_9/gamma* 
_output_shapes
: : 
Ű
2batch_normalization_9/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id* 
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_9/beta
Ů
+batch_normalization_9/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_9/cond/FusedBatchNorm_1/Switch4batch_normalization_9/cond/FusedBatchNorm_1/Switch_14batch_normalization_9/cond/FusedBatchNorm_1/Switch_24batch_normalization_9/cond/FusedBatchNorm_1/Switch_34batch_normalization_9/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
is_training( *
epsilon%o:
ŕ
2batch_normalization_9/cond/FusedBatchNorm_1/SwitchSwitchConv2D_9"batch_normalization_9/cond/pred_id*
T0*
_class
loc:@Conv2D_9*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
ß
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_9/gamma* 
_output_shapes
: : 
Ý
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_9/beta* 
_output_shapes
: : 
ë
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_9/moving_mean/read"batch_normalization_9/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean* 
_output_shapes
: : 
ó
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_9/moving_variance/read"batch_normalization_9/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance* 
_output_shapes
: : 
Č
 batch_normalization_9/cond/MergeMerge+batch_normalization_9/cond/FusedBatchNorm_1)batch_normalization_9/cond/FusedBatchNorm*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 
ˇ
"batch_normalization_9/cond/Merge_1Merge-batch_normalization_9/cond/FusedBatchNorm_1:1+batch_normalization_9/cond/FusedBatchNorm:1*
N*
_output_shapes

: : *
T0
ˇ
"batch_normalization_9/cond/Merge_2Merge-batch_normalization_9/cond/FusedBatchNorm_1:2+batch_normalization_9/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

: : 
n
#batch_normalization_9/cond_1/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

{
%batch_normalization_9/cond_1/switch_tIdentity%batch_normalization_9/cond_1/Switch:1*
_output_shapes
:*
T0

y
%batch_normalization_9/cond_1/switch_fIdentity#batch_normalization_9/cond_1/Switch*
_output_shapes
:*
T0

`
$batch_normalization_9/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_9/cond_1/ConstConst&^batch_normalization_9/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_9/cond_1/Const_1Const&^batch_normalization_9/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Ą
"batch_normalization_9/cond_1/MergeMerge$batch_normalization_9/cond_1/Const_1"batch_normalization_9/cond_1/Const*
_output_shapes
: : *
T0*
N
´
*batch_normalization_9/AssignMovingAvg/readIdentity!batch_normalization_9/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ű
)batch_normalization_9/AssignMovingAvg/SubSub*batch_normalization_9/AssignMovingAvg/read"batch_normalization_9/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ú
)batch_normalization_9/AssignMovingAvg/MulMul)batch_normalization_9/AssignMovingAvg/Sub"batch_normalization_9/cond_1/Merge*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean
î
%batch_normalization_9/AssignMovingAvg	AssignSub!batch_normalization_9/moving_mean)batch_normalization_9/AssignMovingAvg/Mul*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean
ž
,batch_normalization_9/AssignMovingAvg_1/readIdentity%batch_normalization_9/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes
: 
ă
+batch_normalization_9/AssignMovingAvg_1/SubSub,batch_normalization_9/AssignMovingAvg_1/read"batch_normalization_9/cond/Merge_2*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance
â
+batch_normalization_9/AssignMovingAvg_1/MulMul+batch_normalization_9/AssignMovingAvg_1/Sub"batch_normalization_9/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes
: 
ú
'batch_normalization_9/AssignMovingAvg_1	AssignSub%batch_normalization_9/moving_variance+batch_normalization_9/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes
: 

	BiasAdd_9BiasAdd batch_normalization_9/cond/MergeVariable_19/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*
data_formatNHWC
^
Variable_20/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  pB
o
Variable_20
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ž
Variable_20/AssignAssignVariable_20Variable_20/initial_value*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*
_output_shapes
: 
j
Variable_20/readIdentityVariable_20*
_output_shapes
: *
T0*
_class
loc:@Variable_20
^
Variable_21/initial_valueConst*
valueB
 *   C*
dtype0*
_output_shapes
: 
o
Variable_21
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ž
Variable_21/AssignAssignVariable_21Variable_21/initial_value*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
j
Variable_21/readIdentityVariable_21*
_output_shapes
: *
T0*
_class
loc:@Variable_21
r
truncated_normal_10/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
]
truncated_normal_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
truncated_normal_10/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
#truncated_normal_10/TruncatedNormalTruncatedNormaltruncated_normal_10/shape*

seed *
T0*
dtype0*&
_output_shapes
: *
seed2 

truncated_normal_10/mulMul#truncated_normal_10/TruncatedNormaltruncated_normal_10/stddev*&
_output_shapes
: *
T0
~
truncated_normal_10Addtruncated_normal_10/multruncated_normal_10/mean*
T0*&
_output_shapes
: 

Variable_22
VariableV2*&
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
¸
Variable_22/AssignAssignVariable_22truncated_normal_10*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
: 
z
Variable_22/readIdentityVariable_22*
T0*
_class
loc:@Variable_22*&
_output_shapes
: 
U
Const_10Const*
_output_shapes
:*
valueB*ÍĚĚ=*
dtype0
w
Variable_23
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ą
Variable_23/AssignAssignVariable_23Const_10*
T0*
_class
loc:@Variable_23*
validate_shape(*
_output_shapes
:*
use_locking(
n
Variable_23/readIdentityVariable_23*
T0*
_class
loc:@Variable_23*
_output_shapes
:
Ú
	Conv2D_10Conv2D	BiasAdd_9Variable_22/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0
c
mulMul	Conv2D_10Variable_20/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
addAddmulVariable_21/read*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
subSubyadd*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
U
norm/mulMulsubsub*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
c

norm/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
s
norm/SumSumnorm/mul
norm/Const*&
_output_shapes
:*

Tidx0*
	keep_dims(*
T0
L
	norm/SqrtSqrtnorm/Sum*
T0*&
_output_shapes
:
W
norm/SqueezeSqueeze	norm/Sqrt*
squeeze_dims
 *
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
O
lossScalarSummary	loss/tagsnorm/Squeeze*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
z
!gradients/norm/Squeeze_grad/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
 
#gradients/norm/Squeeze_grad/ReshapeReshapegradients/Fill!gradients/norm/Squeeze_grad/Shape*
T0*
Tshape0*&
_output_shapes
:

!gradients/norm/Sqrt_grad/SqrtGradSqrtGrad	norm/Sqrt#gradients/norm/Squeeze_grad/Reshape*
T0*&
_output_shapes
:
~
%gradients/norm/Sum_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ł
gradients/norm/Sum_grad/ReshapeReshape!gradients/norm/Sqrt_grad/SqrtGrad%gradients/norm/Sum_grad/Reshape/shape*&
_output_shapes
:*
T0*
Tshape0
e
gradients/norm/Sum_grad/ShapeShapenorm/mul*
T0*
out_type0*
_output_shapes
:
˛
gradients/norm/Sum_grad/TileTilegradients/norm/Sum_grad/Reshapegradients/norm/Sum_grad/Shape*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
`
gradients/norm/mul_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
b
gradients/norm/mul_grad/Shape_1Shapesub*
T0*
out_type0*
_output_shapes
:
Ă
-gradients/norm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/norm/mul_grad/Shapegradients/norm/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/norm/mul_grad/mulMulgradients/norm/Sum_grad/Tilesub*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients/norm/mul_grad/SumSumgradients/norm/mul_grad/mul-gradients/norm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
°
gradients/norm/mul_grad/ReshapeReshapegradients/norm/mul_grad/Sumgradients/norm/mul_grad/Shape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

gradients/norm/mul_grad/mul_1Mulsubgradients/norm/Sum_grad/Tile*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
gradients/norm/mul_grad/Sum_1Sumgradients/norm/mul_grad/mul_1/gradients/norm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ś
!gradients/norm/mul_grad/Reshape_1Reshapegradients/norm/mul_grad/Sum_1gradients/norm/mul_grad/Shape_1*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
(gradients/norm/mul_grad/tuple/group_depsNoOp ^gradients/norm/mul_grad/Reshape"^gradients/norm/mul_grad/Reshape_1
ř
0gradients/norm/mul_grad/tuple/control_dependencyIdentitygradients/norm/mul_grad/Reshape)^gradients/norm/mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/norm/mul_grad/Reshape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ţ
2gradients/norm/mul_grad/tuple/control_dependency_1Identity!gradients/norm/mul_grad/Reshape_1)^gradients/norm/mul_grad/tuple/group_deps*4
_class*
(&loc:@gradients/norm/mul_grad/Reshape_1*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ő
gradients/AddNAddN0gradients/norm/mul_grad/tuple/control_dependency2gradients/norm/mul_grad/tuple/control_dependency_1*
T0*2
_class(
&$loc:@gradients/norm/mul_grad/Reshape*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
gradients/sub_grad/ShapeShapey*
out_type0*
_output_shapes
:*
T0
]
gradients/sub_grad/Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/sub_grad/SumSumgradients/AddN(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ą
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/sub_grad/Sum_1Sumgradients/AddN*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ľ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
ä
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
gradients/add_grad/ShapeShapemul*
_output_shapes
:*
T0*
out_type0
]
gradients/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ś
gradients/add_grad/SumSum-gradients/sub_grad/tuple/control_dependency_1(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ą
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients/add_grad/Sum_1Sum-gradients/sub_grad/tuple/control_dependency_1*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
ä
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
a
gradients/mul_grad/ShapeShape	Conv2D_10*
T0*
out_type0*
_output_shapes
:
]
gradients/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
´
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mulMul+gradients/add_grad/tuple/control_dependencyVariable_20/read*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ą
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/mul_1Mul	Conv2D_10+gradients/add_grad/tuple/control_dependency*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
ä
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
_output_shapes
: 

gradients/Conv2D_10_grad/ShapeNShapeN	BiasAdd_9Variable_22/read*
T0*
out_type0*
N* 
_output_shapes
::
w
gradients/Conv2D_10_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
ć
,gradients/Conv2D_10_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_10_grad/ShapeNVariable_22/read+gradients/mul_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC
ź
-gradients/Conv2D_10_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_9gradients/Conv2D_10_grad/Const+gradients/mul_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

)gradients/Conv2D_10_grad/tuple/group_depsNoOp-^gradients/Conv2D_10_grad/Conv2DBackpropInput.^gradients/Conv2D_10_grad/Conv2DBackpropFilter

1gradients/Conv2D_10_grad/tuple/control_dependencyIdentity,gradients/Conv2D_10_grad/Conv2DBackpropInput*^gradients/Conv2D_10_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

3gradients/Conv2D_10_grad/tuple/control_dependency_1Identity-gradients/Conv2D_10_grad/Conv2DBackpropFilter*^gradients/Conv2D_10_grad/tuple/group_deps*@
_class6
42loc:@gradients/Conv2D_10_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
˘
$gradients/BiasAdd_9_grad/BiasAddGradBiasAddGrad1gradients/Conv2D_10_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes
: *
T0

)gradients/BiasAdd_9_grad/tuple/group_depsNoOp2^gradients/Conv2D_10_grad/tuple/control_dependency%^gradients/BiasAdd_9_grad/BiasAddGrad

1gradients/BiasAdd_9_grad/tuple/control_dependencyIdentity1gradients/Conv2D_10_grad/tuple/control_dependency*^gradients/BiasAdd_9_grad/tuple/group_deps*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput
ď
3gradients/BiasAdd_9_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_9_grad/BiasAddGrad*^gradients/BiasAdd_9_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_9_grad/BiasAddGrad*
_output_shapes
: 
´
9gradients/batch_normalization_9/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_9_grad/tuple/control_dependency"batch_normalization_9/cond/pred_id*
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 

@gradients/batch_normalization_9/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_9/cond/Merge_grad/cond_grad
Ď
Hgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_9/cond/Merge_grad/cond_gradA^gradients/batch_normalization_9/cond/Merge_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ó
Jgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_9/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_9/cond/Merge_grad/tuple/group_deps*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput
u
gradients/zeros_like	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:1*
T0*
_output_shapes
: 
w
gradients/zeros_like_1	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:2*
T0*
_output_shapes
: 
w
gradients/zeros_like_2	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:3*
_output_shapes
: *
T0
w
gradients/zeros_like_3	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:4*
_output_shapes
: *
T0

Mgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependency2batch_normalization_9/cond/FusedBatchNorm_1/Switch4batch_normalization_9/cond/FusedBatchNorm_1/Switch_14batch_normalization_9/cond/FusedBatchNorm_1/Switch_34batch_normalization_9/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Ugradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 

Ugradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 
u
gradients/zeros_like_4	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:1*
T0*
_output_shapes
: 
u
gradients/zeros_like_5	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:2*
T0*
_output_shapes
: 
u
gradients/zeros_like_6	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:3*
T0*
_output_shapes
: 
u
gradients/zeros_like_7	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:4*
_output_shapes
: *
T0
ý
Kgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependency_12batch_normalization_9/cond/FusedBatchNorm/Switch:14batch_normalization_9/cond/FusedBatchNorm/Switch_1:1+batch_normalization_9/cond/FusedBatchNorm:3+batch_normalization_9/cond/FusedBatchNorm:4*
data_formatNHWC*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ : : : : *
is_training(*
epsilon%o:*
T0

Igradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
˙
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˙
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
ý
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ą
gradients/SwitchSwitchConv2D_9"batch_normalization_9/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Kgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : *
T0

gradients/Switch_1Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id* 
_output_shapes
: : *
T0
e
gradients/Shape_2Shapegradients/Switch_1:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
|
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
_output_shapes
: *
T0*

index_type0
đ
Mgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
N*
_output_shapes

: : *
T0

gradients/Switch_2Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id*
T0* 
_output_shapes
: : 
e
gradients/Shape_3Shapegradients/Switch_2:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
_output_shapes
: *
T0*

index_type0
đ
Mgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
T0*
N*
_output_shapes

: : 
Ł
gradients/Switch_3SwitchConv2D_9"batch_normalization_9/cond/pred_id*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
˙
Igradients/batch_normalization_9/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_3Qgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 

gradients/Switch_4Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id*
T0* 
_output_shapes
: : 
c
gradients/Shape_5Shapegradients/Switch_4*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_4/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
|
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*

index_type0*
_output_shapes
: 
ě
Kgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_4Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes

: : 

gradients/Switch_5Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id*
T0* 
_output_shapes
: : 
c
gradients/Shape_6Shapegradients/Switch_5*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_5/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
|
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
T0*

index_type0*
_output_shapes
: 
ě
Kgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_5Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
N*
_output_shapes

: : *
T0
Ő
gradients/AddN_1AddNKgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_grad/cond_grad

gradients/Conv2D_9_grad/ShapeNShapeNResizeNearestNeighbor_2Variable_18/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_9_grad/ConstConst*%
valueB"      @       *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_9_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_9_grad/ShapeNVariable_18/readgradients/AddN_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
­
,gradients/Conv2D_9_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighbor_2gradients/Conv2D_9_grad/Constgradients/AddN_1*
paddingSAME*&
_output_shapes
:@ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/Conv2D_9_grad/tuple/group_depsNoOp,^gradients/Conv2D_9_grad/Conv2DBackpropInput-^gradients/Conv2D_9_grad/Conv2DBackpropFilter

0gradients/Conv2D_9_grad/tuple/control_dependencyIdentity+gradients/Conv2D_9_grad/Conv2DBackpropInput)^gradients/Conv2D_9_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_9_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@

2gradients/Conv2D_9_grad/tuple/control_dependency_1Identity,gradients/Conv2D_9_grad/Conv2DBackpropFilter)^gradients/Conv2D_9_grad/tuple/group_deps*&
_output_shapes
:@ *
T0*?
_class5
31loc:@gradients/Conv2D_9_grad/Conv2DBackpropFilter
Ä
gradients/AddN_2AddNMgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_1_grad/cond_grad*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
: *
T0
Ä
gradients/AddN_3AddNMgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
: 

Egradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"Č   Č   *
dtype0*
_output_shapes
:
§
@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad0gradients/Conv2D_9_grad/tuple/control_dependencyEgradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad/size*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
align_corners( 
ą
$gradients/BiasAdd_8_grad/BiasAddGradBiasAddGrad@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*
_output_shapes
:@*
T0*
data_formatNHWC

)gradients/BiasAdd_8_grad/tuple/group_depsNoOpA^gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad%^gradients/BiasAdd_8_grad/BiasAddGrad
ź
1gradients/BiasAdd_8_grad/tuple/control_dependencyIdentity@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*^gradients/BiasAdd_8_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ď
3gradients/BiasAdd_8_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_8_grad/BiasAddGrad*^gradients/BiasAdd_8_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_8_grad/BiasAddGrad*
_output_shapes
:@
Č
9gradients/batch_normalization_8/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_8_grad/tuple/control_dependency"batch_normalization_8/cond/pred_id*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0

@gradients/batch_normalization_8/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_8/cond/Merge_grad/cond_grad
ă
Hgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_8/cond/Merge_grad/cond_gradA^gradients/batch_normalization_8/cond/Merge_grad/tuple/group_deps*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
T0
ç
Jgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_8/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_8/cond/Merge_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
w
gradients/zeros_like_8	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:1*
_output_shapes
:@*
T0
w
gradients/zeros_like_9	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:2*
_output_shapes
:@*
T0
x
gradients/zeros_like_10	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:3*
T0*
_output_shapes
:@
x
gradients/zeros_like_11	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:4*
T0*
_output_shapes
:@

Mgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependency2batch_normalization_8/cond/FusedBatchNorm_1/Switch4batch_normalization_8/cond/FusedBatchNorm_1/Switch_14batch_normalization_8/cond/FusedBatchNorm_1/Switch_34batch_normalization_8/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( 
Ł
Kgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

Ugradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Ugradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@
v
gradients/zeros_like_12	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:1*
T0*
_output_shapes
:@
v
gradients/zeros_like_13	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:2*
T0*
_output_shapes
:@
v
gradients/zeros_like_14	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:3*
_output_shapes
:@*
T0
v
gradients/zeros_like_15	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:4*
_output_shapes
:@*
T0
ý
Kgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependency_12batch_normalization_8/cond/FusedBatchNorm/Switch:14batch_normalization_8/cond/FusedBatchNorm/Switch_1:1+batch_normalization_8/cond/FusedBatchNorm:3+batch_normalization_8/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ČČ@:@:@: : *
is_training(

Igradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
˙
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
˙
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
ý
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ł
gradients/Switch_6SwitchConv2D_8"batch_normalization_8/cond/pred_id*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
e
gradients/Shape_7Shapegradients/Switch_6:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

Kgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 

gradients/Switch_7Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id* 
_output_shapes
:@:@*
T0
e
gradients/Shape_8Shapegradients/Switch_7:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_7/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*

index_type0*
_output_shapes
:@
đ
Mgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
T0*
N*
_output_shapes

:@: 

gradients/Switch_8Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_9Shapegradients/Switch_8:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*

index_type0*
_output_shapes
:@
đ
Mgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
N*
_output_shapes

:@: *
T0
Ł
gradients/Switch_9SwitchConv2D_8"batch_normalization_8/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0
d
gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_9/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
˙
Igradients/batch_normalization_8/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_9Qgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 

gradients/Switch_10Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_11Shapegradients/Switch_10*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*

index_type0*
_output_shapes
:@*
T0
í
Kgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_10Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes

:@: 

gradients/Switch_11Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_12Shapegradients/Switch_11*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_11/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
T0*

index_type0*
_output_shapes
:@
í
Kgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_11Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
N*
_output_shapes

:@: *
T0
Ő
gradients/AddN_4AddNKgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

gradients/Conv2D_8_grad/ShapeNShapeNResizeNearestNeighbor_1Variable_16/read*
N* 
_output_shapes
::*
T0*
out_type0
v
gradients/Conv2D_8_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"         @   
É
+gradients/Conv2D_8_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_8_grad/ShapeNVariable_16/readgradients/AddN_4*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ž
,gradients/Conv2D_8_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighbor_1gradients/Conv2D_8_grad/Constgradients/AddN_4*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_8_grad/tuple/group_depsNoOp,^gradients/Conv2D_8_grad/Conv2DBackpropInput-^gradients/Conv2D_8_grad/Conv2DBackpropFilter

0gradients/Conv2D_8_grad/tuple/control_dependencyIdentity+gradients/Conv2D_8_grad/Conv2DBackpropInput)^gradients/Conv2D_8_grad/tuple/group_deps*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0*>
_class4
20loc:@gradients/Conv2D_8_grad/Conv2DBackpropInput

2gradients/Conv2D_8_grad/tuple/control_dependency_1Identity,gradients/Conv2D_8_grad/Conv2DBackpropFilter)^gradients/Conv2D_8_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_8_grad/Conv2DBackpropFilter*'
_output_shapes
:@
Ä
gradients/AddN_5AddNMgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:@
Ä
gradients/AddN_6AddNMgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:@

Egradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"d   d   *
dtype0*
_output_shapes
:
Ś
@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad0gradients/Conv2D_8_grad/tuple/control_dependencyEgradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad/size*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
align_corners( 
˛
$gradients/BiasAdd_7_grad/BiasAddGradBiasAddGrad@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*
_output_shapes	
:*
T0*
data_formatNHWC

)gradients/BiasAdd_7_grad/tuple/group_depsNoOpA^gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad%^gradients/BiasAdd_7_grad/BiasAddGrad
ť
1gradients/BiasAdd_7_grad/tuple/control_dependencyIdentity@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*^gradients/BiasAdd_7_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
đ
3gradients/BiasAdd_7_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_7_grad/BiasAddGrad*^gradients/BiasAdd_7_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_7_grad/BiasAddGrad*
_output_shapes	
:
Ć
9gradients/batch_normalization_7/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_7_grad/tuple/control_dependency"batch_normalization_7/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad

@gradients/batch_normalization_7/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_7/cond/Merge_grad/cond_grad
â
Hgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_7/cond/Merge_grad/cond_gradA^gradients/batch_normalization_7/cond/Merge_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
ć
Jgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_7/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_7/cond/Merge_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
y
gradients/zeros_like_16	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:1*
T0*
_output_shapes	
:
y
gradients/zeros_like_17	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:2*
_output_shapes	
:*
T0
y
gradients/zeros_like_18	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_19	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependency2batch_normalization_7/cond/FusedBatchNorm_1/Switch4batch_normalization_7/cond/FusedBatchNorm_1/Switch_14batch_normalization_7/cond/FusedBatchNorm_1/Switch_34batch_normalization_7/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( 
Ł
Kgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

Ugradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_20	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:1*
_output_shapes	
:*
T0
w
gradients/zeros_like_21	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:2*
_output_shapes	
:*
T0
w
gradients/zeros_like_22	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_23	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependency_12batch_normalization_7/cond/FusedBatchNorm/Switch:14batch_normalization_7/cond/FusedBatchNorm/Switch_1:1+batch_normalization_7/cond/FusedBatchNorm:3+batch_normalization_7/cond/FusedBatchNorm:4*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙dd::: : *
is_training(*
epsilon%o:*
T0

Igradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:

Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
˘
gradients/Switch_12SwitchConv2D_7"batch_normalization_7/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd*
T0
g
gradients/Shape_13Shapegradients/Switch_12:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_12/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0

Kgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: *
T0*
N

gradients/Switch_13Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_14Shapegradients/Switch_13:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_13/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
T0*
N*
_output_shapes
	:: 

gradients/Switch_14Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_15Shapegradients/Switch_14:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_15SwitchConv2D_7"batch_normalization_7/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd*
T0
e
gradients/Shape_16Shapegradients/Switch_15*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_15/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0*

index_type0
˙
Igradients/batch_normalization_7/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_15Qgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 

gradients/Switch_16Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_17Shapegradients/Switch_16*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
_output_shapes	
:*
T0*

index_type0
î
Kgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_16Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_17Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_18Shapegradients/Switch_17*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_17Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ô
gradients/AddN_7AddNKgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

gradients/Conv2D_7_grad/ShapeNShapeNResizeNearestNeighborVariable_14/read*
N* 
_output_shapes
::*
T0*
out_type0
v
gradients/Conv2D_7_grad/ConstConst*
_output_shapes
:*%
valueB"            *
dtype0
É
+gradients/Conv2D_7_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_7_grad/ShapeNVariable_14/readgradients/AddN_7*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
­
,gradients/Conv2D_7_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighborgradients/Conv2D_7_grad/Constgradients/AddN_7*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/Conv2D_7_grad/tuple/group_depsNoOp,^gradients/Conv2D_7_grad/Conv2DBackpropInput-^gradients/Conv2D_7_grad/Conv2DBackpropFilter

0gradients/Conv2D_7_grad/tuple/control_dependencyIdentity+gradients/Conv2D_7_grad/Conv2DBackpropInput)^gradients/Conv2D_7_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_7_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

2gradients/Conv2D_7_grad/tuple/control_dependency_1Identity,gradients/Conv2D_7_grad/Conv2DBackpropFilter)^gradients/Conv2D_7_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_7_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ĺ
gradients/AddN_8AddNMgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ĺ
gradients/AddN_9AddNMgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:

Cgradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"2   2   *
dtype0*
_output_shapes
:
˘
>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad0gradients/Conv2D_7_grad/tuple/control_dependencyCgradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
°
$gradients/BiasAdd_6_grad/BiasAddGradBiasAddGrad>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
_output_shapes	
:*
T0*
data_formatNHWC

)gradients/BiasAdd_6_grad/tuple/group_depsNoOp?^gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad%^gradients/BiasAdd_6_grad/BiasAddGrad
ˇ
1gradients/BiasAdd_6_grad/tuple/control_dependencyIdentity>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*^gradients/BiasAdd_6_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0
đ
3gradients/BiasAdd_6_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_6_grad/BiasAddGrad*^gradients/BiasAdd_6_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_6_grad/BiasAddGrad*
_output_shapes	
:
Ä
9gradients/batch_normalization_6/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_6_grad/tuple/control_dependency"batch_normalization_6/cond/pred_id*
T0*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22

@gradients/batch_normalization_6/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_6/cond/Merge_grad/cond_grad
ŕ
Hgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_6/cond/Merge_grad/cond_gradA^gradients/batch_normalization_6/cond/Merge_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
ä
Jgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_6/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_6/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad
y
gradients/zeros_like_24	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:1*
T0*
_output_shapes	
:
y
gradients/zeros_like_25	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_26	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_27	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:4*
T0*
_output_shapes	
:

Mgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependency2batch_normalization_6/cond/FusedBatchNorm_1/Switch4batch_normalization_6/cond/FusedBatchNorm_1/Switch_14batch_normalization_6/cond/FusedBatchNorm_1/Switch_34batch_normalization_6/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0

Ugradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:*
T0
w
gradients/zeros_like_28	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_29	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_30	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_31	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependency_12batch_normalization_6/cond/FusedBatchNorm/Switch:14batch_normalization_6/cond/FusedBatchNorm/Switch_1:1+batch_normalization_6/cond/FusedBatchNorm:3+batch_normalization_6/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(

Igradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:*
T0

Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ý
Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_18SwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
g
gradients/Shape_19Shapegradients/Switch_18:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_18Fillgradients/Shape_19gradients/zeros_18/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Kgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_18*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_19Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_20Shapegradients/Switch_19:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_19Fillgradients/Shape_20gradients/zeros_19/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_19*
T0*
N*
_output_shapes
	:: 

gradients/Switch_20Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_21Shapegradients/Switch_20:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_20Fillgradients/Shape_21gradients/zeros_20/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_20*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_21SwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
e
gradients/Shape_22Shapegradients/Switch_21*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_21/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_21Fillgradients/Shape_22gradients/zeros_21/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_6/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_21Qgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_22Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_23Shapegradients/Switch_22*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_22Fillgradients/Shape_23gradients/zeros_22/Const*

index_type0*
_output_shapes	
:*
T0
î
Kgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_22Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_23Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_24Shapegradients/Switch_23*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_23/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_23Fillgradients/Shape_24gradients/zeros_23/Const*
_output_shapes	
:*
T0*

index_type0
î
Kgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_23Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ő
gradients/AddN_10AddNKgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_grad/cond_grad*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0

gradients/Conv2D_6_grad/ShapeNShapeN	BiasAdd_5Variable_12/read* 
_output_shapes
::*
T0*
out_type0*
N
v
gradients/Conv2D_6_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
Ę
+gradients/Conv2D_6_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_6_grad/ShapeNVariable_12/readgradients/AddN_10*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
˘
,gradients/Conv2D_6_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_5gradients/Conv2D_6_grad/Constgradients/AddN_10*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_6_grad/tuple/group_depsNoOp,^gradients/Conv2D_6_grad/Conv2DBackpropInput-^gradients/Conv2D_6_grad/Conv2DBackpropFilter

0gradients/Conv2D_6_grad/tuple/control_dependencyIdentity+gradients/Conv2D_6_grad/Conv2DBackpropInput)^gradients/Conv2D_6_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput

2gradients/Conv2D_6_grad/tuple/control_dependency_1Identity,gradients/Conv2D_6_grad/Conv2DBackpropFilter)^gradients/Conv2D_6_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_6_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ć
gradients/AddN_11AddNMgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N
Ć
gradients/AddN_12AddNMgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_5_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_6_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_5_grad/tuple/group_depsNoOp1^gradients/Conv2D_6_grad/tuple/control_dependency%^gradients/BiasAdd_5_grad/BiasAddGrad

1gradients/BiasAdd_5_grad/tuple/control_dependencyIdentity0gradients/Conv2D_6_grad/tuple/control_dependency*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
đ
3gradients/BiasAdd_5_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_5_grad/BiasAddGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_5_grad/BiasAddGrad*
_output_shapes	
:
ą
9gradients/batch_normalization_5/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_5_grad/tuple/control_dependency"batch_normalization_5/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22

@gradients/batch_normalization_5/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_5/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_5/cond/Merge_grad/cond_gradA^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Ń
Jgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_5/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
y
gradients/zeros_like_32	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:1*
_output_shapes	
:*
T0
y
gradients/zeros_like_33	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:2*
_output_shapes	
:*
T0
y
gradients/zeros_like_34	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_35	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:4*
T0*
_output_shapes	
:

Mgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( 
Ł
Kgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
w
gradients/zeros_like_36	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_37	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_38	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:3*
_output_shapes	
:*
T0
w
gradients/zeros_like_39	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_12batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:1+batch_normalization_5/cond/FusedBatchNorm:3+batch_normalization_5/cond/FusedBatchNorm:4*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(*
epsilon%o:*
T0

Igradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:

Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ý
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_24SwitchConv2D_5"batch_normalization_5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0
g
gradients/Shape_25Shapegradients/Switch_24:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_24/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_24Fillgradients/Shape_25gradients/zeros_24/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Kgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_24*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_25Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_26Shapegradients/Switch_25:1*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_25/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_25Fillgradients/Shape_26gradients/zeros_25/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_25*
N*
_output_shapes
	:: *
T0

gradients/Switch_26Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_27Shapegradients/Switch_26:1*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_26/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_26Fillgradients/Shape_27gradients/zeros_26/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_26*
_output_shapes
	:: *
T0*
N
˘
gradients/Switch_27SwitchConv2D_5"batch_normalization_5/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
e
gradients/Shape_28Shapegradients/Switch_27*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_27/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_27Fillgradients/Shape_28gradients/zeros_27/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_27Qgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_28Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_29Shapegradients/Switch_28*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_28Fillgradients/Shape_29gradients/zeros_28/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_28Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_29Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_30Shapegradients/Switch_29*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_29/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_29Fillgradients/Shape_30gradients/zeros_29/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_29Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
_output_shapes
	:: *
T0*
N
Ő
gradients/AddN_13AddNKgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

gradients/Conv2D_5_grad/ShapeNShapeN	BiasAdd_4Variable_10/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_5_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
Ę
+gradients/Conv2D_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_5_grad/ShapeNVariable_10/readgradients/AddN_13*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
˘
,gradients/Conv2D_5_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_4gradients/Conv2D_5_grad/Constgradients/AddN_13*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:

(gradients/Conv2D_5_grad/tuple/group_depsNoOp,^gradients/Conv2D_5_grad/Conv2DBackpropInput-^gradients/Conv2D_5_grad/Conv2DBackpropFilter

0gradients/Conv2D_5_grad/tuple/control_dependencyIdentity+gradients/Conv2D_5_grad/Conv2DBackpropInput)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

2gradients/Conv2D_5_grad/tuple/control_dependency_1Identity,gradients/Conv2D_5_grad/Conv2DBackpropFilter)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_5_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ć
gradients/AddN_14AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_15AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_5_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_4_grad/tuple/group_depsNoOp1^gradients/Conv2D_5_grad/tuple/control_dependency%^gradients/BiasAdd_4_grad/BiasAddGrad

1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentity0gradients/Conv2D_5_grad/tuple/control_dependency*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
đ
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes	
:
ą
9gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_4_grad/tuple/control_dependency"batch_normalization_4/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22

@gradients/batch_normalization_4/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_4/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_4/cond/Merge_grad/cond_gradA^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Ń
Jgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
y
gradients/zeros_like_40	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:1*
_output_shapes	
:*
T0
y
gradients/zeros_like_41	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:2*
_output_shapes	
:*
T0
y
gradients/zeros_like_42	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_43	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:4*
T0*
_output_shapes	
:

Mgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( 
Ł
Kgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_44	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:1*
_output_shapes	
:*
T0
w
gradients/zeros_like_45	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:2*
_output_shapes	
:*
T0
w
gradients/zeros_like_46	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:3*
_output_shapes	
:*
T0
w
gradients/zeros_like_47	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:4*
_output_shapes	
:*
T0
ţ
Kgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_12batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:1+batch_normalization_4/cond/FusedBatchNorm:3+batch_normalization_4/cond/FusedBatchNorm:4*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(*
epsilon%o:*
T0

Igradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:

Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:*
T0
ý
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_30SwitchConv2D_4"batch_normalization_4/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0
g
gradients/Shape_31Shapegradients/Switch_30:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_30Fillgradients/Shape_31gradients/zeros_30/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Kgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_30*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_31Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_32Shapegradients/Switch_31:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_31/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_31Fillgradients/Shape_32gradients/zeros_31/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_31*
T0*
N*
_output_shapes
	:: 

gradients/Switch_32Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_33Shapegradients/Switch_32:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_32Fillgradients/Shape_33gradients/zeros_32/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_32*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_33SwitchConv2D_4"batch_normalization_4/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0
e
gradients/Shape_34Shapegradients/Switch_33*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_33Fillgradients/Shape_34gradients/zeros_33/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_33Qgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_34Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_35Shapegradients/Switch_34*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_34Fillgradients/Shape_35gradients/zeros_34/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_34Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_35Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_36Shapegradients/Switch_35*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_35Fillgradients/Shape_36gradients/zeros_35/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_35Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ő
gradients/AddN_16AddNKgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_grad

gradients/Conv2D_4_grad/ShapeNShapeN	BiasAdd_3Variable_8/read*
N* 
_output_shapes
::*
T0*
out_type0
v
gradients/Conv2D_4_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/readgradients/AddN_16*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
˘
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_3gradients/Conv2D_4_grad/Constgradients/AddN_16*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter

0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput

2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ć
gradients/AddN_17AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_18AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_4_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_3_grad/tuple/group_depsNoOp1^gradients/Conv2D_4_grad/tuple/control_dependency%^gradients/BiasAdd_3_grad/BiasAddGrad

1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentity0gradients/Conv2D_4_grad/tuple/control_dependency*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
đ
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
_output_shapes	
:*
T0*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad
ą
9gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_3_grad/tuple/control_dependency"batch_normalization_3/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22

@gradients/batch_normalization_3/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_3/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_3/cond/Merge_grad/cond_gradA^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput
Ń
Jgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
y
gradients/zeros_like_48	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:1*
T0*
_output_shapes	
:
y
gradients/zeros_like_49	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_50	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_51	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:4*
T0*
_output_shapes	
:

Mgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:*
T0*
data_formatNHWC
Ł
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_52	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:1*
_output_shapes	
:*
T0
w
gradients/zeros_like_53	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:2*
_output_shapes	
:*
T0
w
gradients/zeros_like_54	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:3*
_output_shapes	
:*
T0
w
gradients/zeros_like_55	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_12batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:1+batch_normalization_3/cond/FusedBatchNorm:3+batch_normalization_3/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(

Igradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:

Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_36SwitchConv2D_3"batch_normalization_3/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
g
gradients/Shape_37Shapegradients/Switch_36:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_36/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_36Fillgradients/Shape_37gradients/zeros_36/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Kgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_36*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_37Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_38Shapegradients/Switch_37:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_37/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_37Fillgradients/Shape_38gradients/zeros_37/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_37*
T0*
N*
_output_shapes
	:: 

gradients/Switch_38Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_39Shapegradients/Switch_38:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_38Fillgradients/Shape_39gradients/zeros_38/Const*
_output_shapes	
:*
T0*

index_type0
ň
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_38*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_39SwitchConv2D_3"batch_normalization_3/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
e
gradients/Shape_40Shapegradients/Switch_39*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_39/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_39Fillgradients/Shape_40gradients/zeros_39/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_39Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_40Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_41Shapegradients/Switch_40*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_40/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_40Fillgradients/Shape_41gradients/zeros_40/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_40Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
N*
_output_shapes
	:: *
T0

gradients/Switch_41Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_42Shapegradients/Switch_41*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_41/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_41Fillgradients/Shape_42gradients/zeros_41/Const*
_output_shapes	
:*
T0*

index_type0
î
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_41Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
N*
_output_shapes
	:: *
T0
Ő
gradients/AddN_19AddNKgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

gradients/Conv2D_3_grad/ShapeNShapeN	BiasAdd_2Variable_6/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_3_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/readgradients/AddN_19*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
˘
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_2gradients/Conv2D_3_grad/Constgradients/AddN_19*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter

0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ć
gradients/AddN_20AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad
Ć
gradients/AddN_21AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_3_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:*
T0

)gradients/BiasAdd_2_grad/tuple/group_depsNoOp1^gradients/Conv2D_3_grad/tuple/control_dependency%^gradients/BiasAdd_2_grad/BiasAddGrad

1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentity0gradients/Conv2D_3_grad/tuple/control_dependency*^gradients/BiasAdd_2_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0
đ
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
ą
9gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_2_grad/tuple/control_dependency"batch_normalization_2/cond/pred_id*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd*
T0

@gradients/batch_normalization_2/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_2/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_2/cond/Merge_grad/cond_gradA^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
Ń
Jgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput
y
gradients/zeros_like_56	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:1*
T0*
_output_shapes	
:
y
gradients/zeros_like_57	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_58	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_59	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0

Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_60	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_61	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:2*
_output_shapes	
:*
T0
w
gradients/zeros_like_62	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:3*
_output_shapes	
:*
T0
w
gradients/zeros_like_63	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:4*
_output_shapes	
:*
T0
ţ
Kgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_12batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:1+batch_normalization_2/cond/FusedBatchNorm:3+batch_normalization_2/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙dd::: : *
is_training(*
epsilon%o:

Igradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:

Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ý
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_42SwitchConv2D_2"batch_normalization_2/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
g
gradients/Shape_43Shapegradients/Switch_42:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_42/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_42Fillgradients/Shape_43gradients/zeros_42/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

Kgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_42*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: *
T0

gradients/Switch_43Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_44Shapegradients/Switch_43:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_43/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_43Fillgradients/Shape_44gradients/zeros_43/Const*
_output_shapes	
:*
T0*

index_type0
ň
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_43*
T0*
N*
_output_shapes
	:: 

gradients/Switch_44Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_45Shapegradients/Switch_44:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_44/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_44Fillgradients/Shape_45gradients/zeros_44/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_44*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_45SwitchConv2D_2"batch_normalization_2/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
e
gradients/Shape_46Shapegradients/Switch_45*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_45/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_45Fillgradients/Shape_46gradients/zeros_45/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
˙
Igradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_45Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 

gradients/Switch_46Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*"
_output_shapes
::*
T0
e
gradients/Shape_47Shapegradients/Switch_46*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_46/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_46Fillgradients/Shape_47gradients/zeros_46/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_46Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_47Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_48Shapegradients/Switch_47*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_47/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_47Fillgradients/Shape_48gradients/zeros_47/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_47Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ő
gradients/AddN_22AddNKgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

gradients/Conv2D_2_grad/ShapeNShapeN	BiasAdd_1Variable_4/read* 
_output_shapes
::*
T0*
out_type0*
N
v
gradients/Conv2D_2_grad/ConstConst*
_output_shapes
:*%
valueB"            *
dtype0
É
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/readgradients/AddN_22*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0
˘
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_1gradients/Conv2D_2_grad/Constgradients/AddN_22*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0
Ć
gradients/AddN_23AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_24AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_grad*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:*
T0
˘
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_2_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_1_grad/tuple/group_depsNoOp1^gradients/Conv2D_2_grad/tuple/control_dependency%^gradients/BiasAdd_1_grad/BiasAddGrad

1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentity0gradients/Conv2D_2_grad/tuple/control_dependency*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ
đ
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad
ľ
9gradients/batch_normalization_1/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_1_grad/tuple/control_dependency"batch_normalization_1/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ

@gradients/batch_normalization_1/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_1/cond/Merge_grad/cond_grad
Ď
Hgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_1/cond/Merge_grad/cond_gradA^gradients/batch_normalization_1/cond/Merge_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0
Ó
Jgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_1/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_1/cond/Merge_grad/tuple/group_deps*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
y
gradients/zeros_like_64	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:1*
_output_shapes	
:*
T0
y
gradients/zeros_like_65	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:2*
_output_shapes	
:*
T0
y
gradients/zeros_like_66	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:3*
_output_shapes	
:*
T0
y
gradients/zeros_like_67	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependency2batch_normalization_1/cond/FusedBatchNorm_1/Switch4batch_normalization_1/cond/FusedBatchNorm_1/Switch_14batch_normalization_1/cond/FusedBatchNorm_1/Switch_34batch_normalization_1/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ::::*
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

Ugradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Ugradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
w
gradients/zeros_like_68	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_69	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_70	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_71	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:

Kgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependency_12batch_normalization_1/cond/FusedBatchNorm/Switch:14batch_normalization_1/cond/FusedBatchNorm/Switch_1:1+batch_normalization_1/cond/FusedBatchNorm:3+batch_normalization_1/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*H
_output_shapes6
4:˙˙˙˙˙˙˙˙˙ČČ::: : *
is_training(

Igradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:*
T0

Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ś
gradients/Switch_48SwitchConv2D_1"batch_normalization_1/cond/pred_id*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ*
T0
g
gradients/Shape_49Shapegradients/Switch_48:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_48/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_48Fillgradients/Shape_49gradients/zeros_48/Const*
T0*

index_type0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

Kgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_48*
T0*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙ČČ: 

gradients/Switch_49Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_50Shapegradients/Switch_49:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_49/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_49Fillgradients/Shape_50gradients/zeros_49/Const*

index_type0*
_output_shapes	
:*
T0
ň
Mgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_49*
T0*
N*
_output_shapes
	:: 

gradients/Switch_50Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_51Shapegradients/Switch_50:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_50/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_50Fillgradients/Shape_51gradients/zeros_50/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_50*
T0*
N*
_output_shapes
	:: 
Ś
gradients/Switch_51SwitchConv2D_1"batch_normalization_1/cond/pred_id*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ*
T0
e
gradients/Shape_52Shapegradients/Switch_51*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_51/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_51Fillgradients/Shape_52gradients/zeros_51/Const*
T0*

index_type0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

Igradients/batch_normalization_1/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_51Qgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙ČČ: *
T0

gradients/Switch_52Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*"
_output_shapes
::*
T0
e
gradients/Shape_53Shapegradients/Switch_52*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_52/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_52Fillgradients/Shape_53gradients/zeros_52/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_52Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
N*
_output_shapes
	:: *
T0

gradients/Switch_53Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_54Shapegradients/Switch_53*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_53/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_53Fillgradients/Shape_54gradients/zeros_53/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_53Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
×
gradients/AddN_25AddNKgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_grad/cond_grad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N

gradients/Conv2D_1_grad/ShapeNShapeNBiasAddVariable_2/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_1_grad/ConstConst*%
valueB"      @      *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/AddN_25*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0

,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterBiasAddgradients/Conv2D_1_grad/Constgradients/AddN_25*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*'
_output_shapes
:@*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
Ć
gradients/AddN_26AddNMgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_27AddNMgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:

"gradients/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes
:@*
T0

'gradients/BiasAdd_grad/tuple/group_depsNoOp1^gradients/Conv2D_1_grad/tuple/control_dependency#^gradients/BiasAdd_grad/BiasAddGrad

/gradients/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/Conv2D_1_grad/tuple/control_dependency(^gradients/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ç
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
­
7gradients/batch_normalization/cond/Merge_grad/cond_gradSwitch/gradients/BiasAdd_grad/tuple/control_dependency batch_normalization/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@

>gradients/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp8^gradients/batch_normalization/cond/Merge_grad/cond_grad
Č
Fgradients/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentity7gradients/batch_normalization/cond/Merge_grad/cond_grad?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
Ě
Hgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_1Identity9gradients/batch_normalization/cond/Merge_grad/cond_grad:1?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
v
gradients/zeros_like_72	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:1*
T0*
_output_shapes
:@
v
gradients/zeros_like_73	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:2*
T0*
_output_shapes
:@
v
gradients/zeros_like_74	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:3*
T0*
_output_shapes
:@
v
gradients/zeros_like_75	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:4*
T0*
_output_shapes
:@

Kgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradFgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( *
epsilon%o:

Igradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpL^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Qgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityKgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradJ^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
˙
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@
˙
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
_output_shapes
:@*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
t
gradients/zeros_like_76	ZerosLike)batch_normalization/cond/FusedBatchNorm:1*
_output_shapes
:@*
T0
t
gradients/zeros_like_77	ZerosLike)batch_normalization/cond/FusedBatchNorm:2*
_output_shapes
:@*
T0
t
gradients/zeros_like_78	ZerosLike)batch_normalization/cond/FusedBatchNorm:3*
_output_shapes
:@*
T0
t
gradients/zeros_like_79	ZerosLike)batch_normalization/cond/FusedBatchNorm:4*
T0*
_output_shapes
:@
ń
Igradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_10batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:1)batch_normalization/cond/FusedBatchNorm:3)batch_normalization/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ČČ@:@:@: : *
is_training(

Ggradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOpJ^gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityIgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradH^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
÷
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
÷
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
ő
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ő
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
 
gradients/Switch_54SwitchConv2D batch_normalization/cond/pred_id*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
g
gradients/Shape_55Shapegradients/Switch_54:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_54/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_54Fillgradients/Shape_55gradients/zeros_54/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

Igradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_54*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 

gradients/Switch_55Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0
g
gradients/Shape_56Shapegradients/Switch_55:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_55/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_55Fillgradients/Shape_56gradients/zeros_55/Const*
T0*

index_type0*
_output_shapes
:@
í
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_55*
T0*
N*
_output_shapes

:@: 

gradients/Switch_56Switchbatch_normalization/beta/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0
g
gradients/Shape_57Shapegradients/Switch_56:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_56/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_56Fillgradients/Shape_57gradients/zeros_56/Const*

index_type0*
_output_shapes
:@*
T0
í
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_56*
T0*
N*
_output_shapes

:@: 
 
gradients/Switch_57SwitchConv2D batch_normalization/cond/pred_id*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
e
gradients/Shape_58Shapegradients/Switch_57*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_57/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_57Fillgradients/Shape_58gradients/zeros_57/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ü
Ggradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_57Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 

gradients/Switch_58Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_59Shapegradients/Switch_58*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_58/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_58Fillgradients/Shape_59gradients/zeros_58/Const*
T0*

index_type0*
_output_shapes
:@
é
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_58Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
N*
_output_shapes

:@: *
T0

gradients/Switch_59Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_60Shapegradients/Switch_59*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_59/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_59Fillgradients/Shape_60gradients/zeros_59/Const*
_output_shapes
:@*
T0*

index_type0
é
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_59Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes

:@: 
Đ
gradients/AddN_28AddNIgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradGgradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
|
gradients/Conv2D_grad/ShapeNShapeNxVariable/read*
T0*
out_type0*
N* 
_output_shapes
::
t
gradients/Conv2D_grad/ConstConst*
_output_shapes
:*%
valueB"         @   *
dtype0
Ă
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/AddN_28*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterxgradients/Conv2D_grad/Constgradients/AddN_28*&
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@
ż
gradients/AddN_29AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:@
ż
gradients/AddN_30AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:@*
T0
Š
3Variable/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"         @   *
_class
loc:@Variable*
dtype0*
_output_shapes
:

)Variable/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable*
dtype0*
_output_shapes
: 
ë
#Variable/Adadelta/Initializer/zerosFill3Variable/Adadelta/Initializer/zeros/shape_as_tensor)Variable/Adadelta/Initializer/zeros/Const*&
_output_shapes
:@*
T0*

index_type0*
_class
loc:@Variable
˛
Variable/Adadelta
VariableV2*
dtype0*&
_output_shapes
:@*
shared_name *
_class
loc:@Variable*
	container *
shape:@
Ń
Variable/Adadelta/AssignAssignVariable/Adadelta#Variable/Adadelta/Initializer/zeros*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable

Variable/Adadelta/readIdentityVariable/Adadelta*
T0*
_class
loc:@Variable*&
_output_shapes
:@
Ť
5Variable/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         @   *
_class
loc:@Variable

+Variable/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable
ń
%Variable/Adadelta_1/Initializer/zerosFill5Variable/Adadelta_1/Initializer/zeros/shape_as_tensor+Variable/Adadelta_1/Initializer/zeros/Const*&
_output_shapes
:@*
T0*

index_type0*
_class
loc:@Variable
´
Variable/Adadelta_1
VariableV2*
_class
loc:@Variable*
	container *
shape:@*
dtype0*&
_output_shapes
:@*
shared_name 
×
Variable/Adadelta_1/AssignAssignVariable/Adadelta_1%Variable/Adadelta_1/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(

Variable/Adadelta_1/readIdentityVariable/Adadelta_1*&
_output_shapes
:@*
T0*
_class
loc:@Variable

5Variable_1/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@Variable_1*
dtype0*
_output_shapes
:

+Variable_1/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
ç
%Variable_1/Adadelta/Initializer/zerosFill5Variable_1/Adadelta/Initializer/zeros/shape_as_tensor+Variable_1/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_1*
_output_shapes
:@

Variable_1/Adadelta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Variable_1*
	container *
shape:@
Í
Variable_1/Adadelta/AssignAssignVariable_1/Adadelta%Variable_1/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:@
}
Variable_1/Adadelta/readIdentityVariable_1/Adadelta*
T0*
_class
loc:@Variable_1*
_output_shapes
:@
 
7Variable_1/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@Variable_1*
dtype0*
_output_shapes
:

-Variable_1/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
í
'Variable_1/Adadelta_1/Initializer/zerosFill7Variable_1/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_1/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_1*
_output_shapes
:@
 
Variable_1/Adadelta_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Variable_1*
	container 
Ó
Variable_1/Adadelta_1/AssignAssignVariable_1/Adadelta_1'Variable_1/Adadelta_1/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(

Variable_1/Adadelta_1/readIdentityVariable_1/Adadelta_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:@
ź
Dbatch_normalization/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
:
­
:batch_normalization/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
: 
Ł
4batch_normalization/gamma/Adadelta/Initializer/zerosFillDbatch_normalization/gamma/Adadelta/Initializer/zeros/shape_as_tensor:batch_normalization/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
ź
"batch_normalization/gamma/Adadelta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@

)batch_normalization/gamma/Adadelta/AssignAssign"batch_normalization/gamma/Adadelta4batch_normalization/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@
Ş
'batch_normalization/gamma/Adadelta/readIdentity"batch_normalization/gamma/Adadelta*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
ž
Fbatch_normalization/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
:
Ż
<batch_normalization/gamma/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@batch_normalization/gamma*
dtype0
Š
6batch_normalization/gamma/Adadelta_1/Initializer/zerosFillFbatch_normalization/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor<batch_normalization/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*,
_class"
 loc:@batch_normalization/gamma
ž
$batch_normalization/gamma/Adadelta_1
VariableV2*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@

+batch_normalization/gamma/Adadelta_1/AssignAssign$batch_normalization/gamma/Adadelta_16batch_normalization/gamma/Adadelta_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(
Ž
)batch_normalization/gamma/Adadelta_1/readIdentity$batch_normalization/gamma/Adadelta_1*
_output_shapes
:@*
T0*,
_class"
 loc:@batch_normalization/gamma
ş
Cbatch_normalization/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
Ť
9batch_normalization/beta/Adadelta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@batch_normalization/beta

3batch_normalization/beta/Adadelta/Initializer/zerosFillCbatch_normalization/beta/Adadelta/Initializer/zeros/shape_as_tensor9batch_normalization/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ş
!batch_normalization/beta/Adadelta
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container 

(batch_normalization/beta/Adadelta/AssignAssign!batch_normalization/beta/Adadelta3batch_normalization/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@
§
&batch_normalization/beta/Adadelta/readIdentity!batch_normalization/beta/Adadelta*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ź
Ebatch_normalization/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
­
;batch_normalization/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
: 
Ľ
5batch_normalization/beta/Adadelta_1/Initializer/zerosFillEbatch_normalization/beta/Adadelta_1/Initializer/zeros/shape_as_tensor;batch_normalization/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ź
#batch_normalization/beta/Adadelta_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta

*batch_normalization/beta/Adadelta_1/AssignAssign#batch_normalization/beta/Adadelta_15batch_normalization/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@
Ť
(batch_normalization/beta/Adadelta_1/readIdentity#batch_normalization/beta/Adadelta_1*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
­
5Variable_2/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"      @      *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:

+Variable_2/Adadelta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_2
ô
%Variable_2/Adadelta/Initializer/zerosFill5Variable_2/Adadelta/Initializer/zeros/shape_as_tensor+Variable_2/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*'
_output_shapes
:@
¸
Variable_2/Adadelta
VariableV2*'
_output_shapes
:@*
shared_name *
_class
loc:@Variable_2*
	container *
shape:@*
dtype0
Ú
Variable_2/Adadelta/AssignAssignVariable_2/Adadelta%Variable_2/Adadelta/Initializer/zeros*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_2

Variable_2/Adadelta/readIdentityVariable_2/Adadelta*
T0*
_class
loc:@Variable_2*'
_output_shapes
:@
Ż
7Variable_2/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @      *
_class
loc:@Variable_2

-Variable_2/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_2*
dtype0
ú
'Variable_2/Adadelta_1/Initializer/zerosFill7Variable_2/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_2/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*'
_output_shapes
:@
ş
Variable_2/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape:@*
dtype0*'
_output_shapes
:@
ŕ
Variable_2/Adadelta_1/AssignAssignVariable_2/Adadelta_1'Variable_2/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*'
_output_shapes
:@

Variable_2/Adadelta_1/readIdentityVariable_2/Adadelta_1*
T0*
_class
loc:@Variable_2*'
_output_shapes
:@

5Variable_3/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

+Variable_3/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
č
%Variable_3/Adadelta/Initializer/zerosFill5Variable_3/Adadelta/Initializer/zeros/shape_as_tensor+Variable_3/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_3*
_output_shapes	
:
 
Variable_3/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_3*
	container 
Î
Variable_3/Adadelta/AssignAssignVariable_3/Adadelta%Variable_3/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
~
Variable_3/Adadelta/readIdentityVariable_3/Adadelta*
T0*
_class
loc:@Variable_3*
_output_shapes	
:
Ą
7Variable_3/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

-Variable_3/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
î
'Variable_3/Adadelta_1/Initializer/zerosFill7Variable_3/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_3/Adadelta_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_3*
_output_shapes	
:*
T0
˘
Variable_3/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_3*
	container 
Ô
Variable_3/Adadelta_1/AssignAssignVariable_3/Adadelta_1'Variable_3/Adadelta_1/Initializer/zeros*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

Variable_3/Adadelta_1/readIdentityVariable_3/Adadelta_1*
T0*
_class
loc:@Variable_3*
_output_shapes	
:
Á
Fbatch_normalization_1/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_1/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_1/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_1/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_1/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:
Â
$batch_normalization_1/gamma/Adadelta
VariableV2*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:*
dtype0

+batch_normalization_1/gamma/Adadelta/AssignAssign$batch_normalization_1/gamma/Adadelta6batch_normalization_1/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_1/gamma/Adadelta/readIdentity$batch_normalization_1/gamma/Adadelta*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
T0
Ă
Hbatch_normalization_1/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_1/gamma/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma
˛
8batch_normalization_1/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_1/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_1/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma
Ä
&batch_normalization_1/gamma/Adadelta_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma

-batch_normalization_1/gamma/Adadelta_1/AssignAssign&batch_normalization_1/gamma/Adadelta_18batch_normalization_1/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_1/gamma/Adadelta_1/readIdentity&batch_normalization_1/gamma/Adadelta_1*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
T0
ż
Ebatch_normalization_1/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_1/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_1/beta/Adadelta/Initializer/zerosFillEbatch_normalization_1/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_1/beta/Adadelta/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta
Ŕ
#batch_normalization_1/beta/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:

*batch_normalization_1/beta/Adadelta/AssignAssign#batch_normalization_1/beta/Adadelta5batch_normalization_1/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_1/beta/Adadelta/readIdentity#batch_normalization_1/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
Á
Gbatch_normalization_1/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
ą
=batch_normalization_1/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_1/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_1/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_1/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
Â
%batch_normalization_1/beta/Adadelta_1
VariableV2*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

,batch_normalization_1/beta/Adadelta_1/AssignAssign%batch_normalization_1/beta/Adadelta_17batch_normalization_1/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_1/beta/Adadelta_1/readIdentity%batch_normalization_1/beta/Adadelta_1*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:*
T0
­
5Variable_4/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_4*
dtype0*
_output_shapes
:

+Variable_4/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
ő
%Variable_4/Adadelta/Initializer/zerosFill5Variable_4/Adadelta/Initializer/zeros/shape_as_tensor+Variable_4/Adadelta/Initializer/zeros/Const*(
_output_shapes
:*
T0*

index_type0*
_class
loc:@Variable_4
ş
Variable_4/Adadelta
VariableV2*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_4*
	container *
shape:
Ű
Variable_4/Adadelta/AssignAssignVariable_4/Adadelta%Variable_4/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*(
_output_shapes
:

Variable_4/Adadelta/readIdentityVariable_4/Adadelta*
T0*
_class
loc:@Variable_4*(
_output_shapes
:
Ż
7Variable_4/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_4*
dtype0*
_output_shapes
:

-Variable_4/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
ű
'Variable_4/Adadelta_1/Initializer/zerosFill7Variable_4/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_4/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_4*(
_output_shapes
:
ź
Variable_4/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape:*
dtype0*(
_output_shapes
:
á
Variable_4/Adadelta_1/AssignAssignVariable_4/Adadelta_1'Variable_4/Adadelta_1/Initializer/zeros*
_class
loc:@Variable_4*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

Variable_4/Adadelta_1/readIdentityVariable_4/Adadelta_1*
T0*
_class
loc:@Variable_4*(
_output_shapes
:

5Variable_5/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*
_class
loc:@Variable_5

+Variable_5/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
č
%Variable_5/Adadelta/Initializer/zerosFill5Variable_5/Adadelta/Initializer/zeros/shape_as_tensor+Variable_5/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_5*
_output_shapes	
:
 
Variable_5/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_5*
	container 
Î
Variable_5/Adadelta/AssignAssignVariable_5/Adadelta%Variable_5/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
~
Variable_5/Adadelta/readIdentityVariable_5/Adadelta*
T0*
_class
loc:@Variable_5*
_output_shapes	
:
Ą
7Variable_5/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

-Variable_5/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
î
'Variable_5/Adadelta_1/Initializer/zerosFill7Variable_5/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_5/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_5*
_output_shapes	
:
˘
Variable_5/Adadelta_1
VariableV2*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_5*
	container *
shape:*
dtype0
Ô
Variable_5/Adadelta_1/AssignAssignVariable_5/Adadelta_1'Variable_5/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:

Variable_5/Adadelta_1/readIdentityVariable_5/Adadelta_1*
_class
loc:@Variable_5*
_output_shapes	
:*
T0
Á
Fbatch_normalization_2/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_2/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_2/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_2/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_2/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:
Â
$batch_normalization_2/gamma/Adadelta
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

+batch_normalization_2/gamma/Adadelta/AssignAssign$batch_normalization_2/gamma/Adadelta6batch_normalization_2/gamma/Adadelta/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ą
)batch_normalization_2/gamma/Adadelta/readIdentity$batch_normalization_2/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_2/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0
ł
>batch_normalization_2/gamma/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_2/gamma
˛
8batch_normalization_2/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_2/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_2/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma
Ä
&batch_normalization_2/gamma/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container 

-batch_normalization_2/gamma/Adadelta_1/AssignAssign&batch_normalization_2/gamma/Adadelta_18batch_normalization_2/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_2/gamma/Adadelta_1/readIdentity&batch_normalization_2/gamma/Adadelta_1*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:*
T0
ż
Ebatch_normalization_2/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_2/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_2/beta/Adadelta/Initializer/zerosFillEbatch_normalization_2/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_2/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
Ŕ
#batch_normalization_2/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

*batch_normalization_2/beta/Adadelta/AssignAssign#batch_normalization_2/beta/Adadelta5batch_normalization_2/beta/Adadelta/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(
Ž
(batch_normalization_2/beta/Adadelta/readIdentity#batch_normalization_2/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
Á
Gbatch_normalization_2/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_2/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_2/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_2/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_2/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
Â
%batch_normalization_2/beta/Adadelta_1
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

,batch_normalization_2/beta/Adadelta_1/AssignAssign%batch_normalization_2/beta/Adadelta_17batch_normalization_2/beta/Adadelta_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(
˛
*batch_normalization_2/beta/Adadelta_1/readIdentity%batch_normalization_2/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
­
5Variable_6/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            *
_class
loc:@Variable_6

+Variable_6/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
: 
ő
%Variable_6/Adadelta/Initializer/zerosFill5Variable_6/Adadelta/Initializer/zeros/shape_as_tensor+Variable_6/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_6*(
_output_shapes
:
ş
Variable_6/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape:*
dtype0*(
_output_shapes
:
Ű
Variable_6/Adadelta/AssignAssignVariable_6/Adadelta%Variable_6/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*(
_output_shapes
:

Variable_6/Adadelta/readIdentityVariable_6/Adadelta*
T0*
_class
loc:@Variable_6*(
_output_shapes
:
Ż
7Variable_6/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_6*
dtype0*
_output_shapes
:

-Variable_6/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_6*
dtype0
ű
'Variable_6/Adadelta_1/Initializer/zerosFill7Variable_6/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_6/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_6*(
_output_shapes
:
ź
Variable_6/Adadelta_1
VariableV2*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_6
á
Variable_6/Adadelta_1/AssignAssignVariable_6/Adadelta_1'Variable_6/Adadelta_1/Initializer/zeros*
_class
loc:@Variable_6*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

Variable_6/Adadelta_1/readIdentityVariable_6/Adadelta_1*
T0*
_class
loc:@Variable_6*(
_output_shapes
:

5Variable_7/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

+Variable_7/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
č
%Variable_7/Adadelta/Initializer/zerosFill5Variable_7/Adadelta/Initializer/zeros/shape_as_tensor+Variable_7/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_7*
_output_shapes	
:
 
Variable_7/Adadelta
VariableV2*
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Î
Variable_7/Adadelta/AssignAssignVariable_7/Adadelta%Variable_7/Adadelta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_7
~
Variable_7/Adadelta/readIdentityVariable_7/Adadelta*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
Ą
7Variable_7/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

-Variable_7/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_7
î
'Variable_7/Adadelta_1/Initializer/zerosFill7Variable_7/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_7/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_7*
_output_shapes	
:
˘
Variable_7/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_7*
	container *
shape:
Ô
Variable_7/Adadelta_1/AssignAssignVariable_7/Adadelta_1'Variable_7/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:

Variable_7/Adadelta_1/readIdentityVariable_7/Adadelta_1*
_output_shapes	
:*
T0*
_class
loc:@Variable_7
Á
Fbatch_normalization_3/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_3/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_3/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_3/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_3/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
Â
$batch_normalization_3/gamma/Adadelta
VariableV2*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:*
dtype0

+batch_normalization_3/gamma/Adadelta/AssignAssign$batch_normalization_3/gamma/Adadelta6batch_normalization_3/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_3/gamma/Adadelta/readIdentity$batch_normalization_3/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_3/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0
ł
>batch_normalization_3/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_3/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_3/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_3/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
Ä
&batch_normalization_3/gamma/Adadelta_1
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

-batch_normalization_3/gamma/Adadelta_1/AssignAssign&batch_normalization_3/gamma/Adadelta_18batch_normalization_3/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_3/gamma/Adadelta_1/readIdentity&batch_normalization_3/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
ż
Ebatch_normalization_3/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_3/beta/Adadelta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta
¨
5batch_normalization_3/beta/Adadelta/Initializer/zerosFillEbatch_normalization_3/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_3/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
Ŕ
#batch_normalization_3/beta/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:

*batch_normalization_3/beta/Adadelta/AssignAssign#batch_normalization_3/beta/Adadelta5batch_normalization_3/beta/Adadelta/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(
Ž
(batch_normalization_3/beta/Adadelta/readIdentity#batch_normalization_3/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
Á
Gbatch_normalization_3/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_3/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_3/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_3/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_3/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
Â
%batch_normalization_3/beta/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:

,batch_normalization_3/beta/Adadelta_1/AssignAssign%batch_normalization_3/beta/Adadelta_17batch_normalization_3/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_3/beta/Adadelta_1/readIdentity%batch_normalization_3/beta/Adadelta_1*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_3/beta
­
5Variable_8/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:

+Variable_8/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_8*
dtype0*
_output_shapes
: 
ő
%Variable_8/Adadelta/Initializer/zerosFill5Variable_8/Adadelta/Initializer/zeros/shape_as_tensor+Variable_8/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_8*(
_output_shapes
:
ş
Variable_8/Adadelta
VariableV2*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_8*
	container *
shape:*
dtype0
Ű
Variable_8/Adadelta/AssignAssignVariable_8/Adadelta%Variable_8/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:

Variable_8/Adadelta/readIdentityVariable_8/Adadelta*
T0*
_class
loc:@Variable_8*(
_output_shapes
:
Ż
7Variable_8/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:

-Variable_8/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_8
ű
'Variable_8/Adadelta_1/Initializer/zerosFill7Variable_8/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_8/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_8*(
_output_shapes
:
ź
Variable_8/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_8*
	container *
shape:*
dtype0*(
_output_shapes
:
á
Variable_8/Adadelta_1/AssignAssignVariable_8/Adadelta_1'Variable_8/Adadelta_1/Initializer/zeros*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

Variable_8/Adadelta_1/readIdentityVariable_8/Adadelta_1*
T0*
_class
loc:@Variable_8*(
_output_shapes
:

5Variable_9/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_9*
dtype0*
_output_shapes
:

+Variable_9/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
: 
č
%Variable_9/Adadelta/Initializer/zerosFill5Variable_9/Adadelta/Initializer/zeros/shape_as_tensor+Variable_9/Adadelta/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_9*
_output_shapes	
:*
T0
 
Variable_9/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_9*
	container *
shape:*
dtype0*
_output_shapes	
:
Î
Variable_9/Adadelta/AssignAssignVariable_9/Adadelta%Variable_9/Adadelta/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(
~
Variable_9/Adadelta/readIdentityVariable_9/Adadelta*
_output_shapes	
:*
T0*
_class
loc:@Variable_9
Ą
7Variable_9/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_9*
dtype0*
_output_shapes
:

-Variable_9/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
: 
î
'Variable_9/Adadelta_1/Initializer/zerosFill7Variable_9/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_9/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_9*
_output_shapes	
:
˘
Variable_9/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_9*
	container 
Ô
Variable_9/Adadelta_1/AssignAssignVariable_9/Adadelta_1'Variable_9/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:

Variable_9/Adadelta_1/readIdentityVariable_9/Adadelta_1*
T0*
_class
loc:@Variable_9*
_output_shapes	
:
Á
Fbatch_normalization_4/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_4/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_4/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_4/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_4/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
Â
$batch_normalization_4/gamma/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:

+batch_normalization_4/gamma/Adadelta/AssignAssign$batch_normalization_4/gamma/Adadelta6batch_normalization_4/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_4/gamma/Adadelta/readIdentity$batch_normalization_4/gamma/Adadelta*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_4/gamma
Ă
Hbatch_normalization_4/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*.
_class$
" loc:@batch_normalization_4/gamma
ł
>batch_normalization_4/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_4/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_4/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_4/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
Ä
&batch_normalization_4/gamma/Adadelta_1
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

-batch_normalization_4/gamma/Adadelta_1/AssignAssign&batch_normalization_4/gamma/Adadelta_18batch_normalization_4/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_4/gamma/Adadelta_1/readIdentity&batch_normalization_4/gamma/Adadelta_1*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:*
T0
ż
Ebatch_normalization_4/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_4/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_4/beta/Adadelta/Initializer/zerosFillEbatch_normalization_4/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_4/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
Ŕ
#batch_normalization_4/beta/Adadelta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_4/beta

*batch_normalization_4/beta/Adadelta/AssignAssign#batch_normalization_4/beta/Adadelta5batch_normalization_4/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_4/beta/Adadelta/readIdentity#batch_normalization_4/beta/Adadelta*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:*
T0
Á
Gbatch_normalization_4/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_4/beta*
dtype0
ą
=batch_normalization_4/beta/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0
Ž
7batch_normalization_4/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_4/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_4/beta/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta
Â
%batch_normalization_4/beta/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_4/beta*
	container 

,batch_normalization_4/beta/Adadelta_1/AssignAssign%batch_normalization_4/beta/Adadelta_17batch_normalization_4/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_4/beta/Adadelta_1/readIdentity%batch_normalization_4/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
Ż
6Variable_10/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

,Variable_10/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_10*
dtype0*
_output_shapes
: 
ů
&Variable_10/Adadelta/Initializer/zerosFill6Variable_10/Adadelta/Initializer/zeros/shape_as_tensor,Variable_10/Adadelta/Initializer/zeros/Const*(
_output_shapes
:*
T0*

index_type0*
_class
loc:@Variable_10
ź
Variable_10/Adadelta
VariableV2*
shape:*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_10*
	container 
ß
Variable_10/Adadelta/AssignAssignVariable_10/Adadelta&Variable_10/Adadelta/Initializer/zeros*
T0*
_class
loc:@Variable_10*
validate_shape(*(
_output_shapes
:*
use_locking(

Variable_10/Adadelta/readIdentityVariable_10/Adadelta*
T0*
_class
loc:@Variable_10*(
_output_shapes
:
ą
8Variable_10/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

.Variable_10/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_10*
dtype0*
_output_shapes
: 
˙
(Variable_10/Adadelta_1/Initializer/zerosFill8Variable_10/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_10/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_10*(
_output_shapes
:
ž
Variable_10/Adadelta_1
VariableV2*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_10*
	container *
shape:
ĺ
Variable_10/Adadelta_1/AssignAssignVariable_10/Adadelta_1(Variable_10/Adadelta_1/Initializer/zeros*
T0*
_class
loc:@Variable_10*
validate_shape(*(
_output_shapes
:*
use_locking(

Variable_10/Adadelta_1/readIdentityVariable_10/Adadelta_1*
T0*
_class
loc:@Variable_10*(
_output_shapes
:
Ą
6Variable_11/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*
_class
loc:@Variable_11

,Variable_11/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_11*
dtype0*
_output_shapes
: 
ě
&Variable_11/Adadelta/Initializer/zerosFill6Variable_11/Adadelta/Initializer/zeros/shape_as_tensor,Variable_11/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_11*
_output_shapes	
:
˘
Variable_11/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_11*
	container 
Ň
Variable_11/Adadelta/AssignAssignVariable_11/Adadelta&Variable_11/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:

Variable_11/Adadelta/readIdentityVariable_11/Adadelta*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
Ł
8Variable_11/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

.Variable_11/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_11*
dtype0*
_output_shapes
: 
ň
(Variable_11/Adadelta_1/Initializer/zerosFill8Variable_11/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_11/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_11*
_output_shapes	
:
¤
Variable_11/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_11*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
Variable_11/Adadelta_1/AssignAssignVariable_11/Adadelta_1(Variable_11/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:

Variable_11/Adadelta_1/readIdentityVariable_11/Adadelta_1*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
Á
Fbatch_normalization_5/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_5/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_5/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_5/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_5/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
Â
$batch_normalization_5/gamma/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma*
	container 

+batch_normalization_5/gamma/Adadelta/AssignAssign$batch_normalization_5/gamma/Adadelta6batch_normalization_5/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_5/gamma/Adadelta/readIdentity$batch_normalization_5/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_5/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_5/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_5/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_5/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_5/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma
Ä
&batch_normalization_5/gamma/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma*
	container *
shape:

-batch_normalization_5/gamma/Adadelta_1/AssignAssign&batch_normalization_5/gamma/Adadelta_18batch_normalization_5/gamma/Adadelta_1/Initializer/zeros*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ľ
+batch_normalization_5/gamma/Adadelta_1/readIdentity&batch_normalization_5/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
ż
Ebatch_normalization_5/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_5/beta*
dtype0
Ż
;batch_normalization_5/beta/Adadelta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta
¨
5batch_normalization_5/beta/Adadelta/Initializer/zerosFillEbatch_normalization_5/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_5/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Ŕ
#batch_normalization_5/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_5/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

*batch_normalization_5/beta/Adadelta/AssignAssign#batch_normalization_5/beta/Adadelta5batch_normalization_5/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_5/beta/Adadelta/readIdentity#batch_normalization_5/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Á
Gbatch_normalization_5/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_5/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_5/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_5/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_5/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Â
%batch_normalization_5/beta/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_5/beta*
	container 

,batch_normalization_5/beta/Adadelta_1/AssignAssign%batch_normalization_5/beta/Adadelta_17batch_normalization_5/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_5/beta/Adadelta_1/readIdentity%batch_normalization_5/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Ż
6Variable_12/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            *
_class
loc:@Variable_12

,Variable_12/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
ů
&Variable_12/Adadelta/Initializer/zerosFill6Variable_12/Adadelta/Initializer/zeros/shape_as_tensor,Variable_12/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_12*(
_output_shapes
:
ź
Variable_12/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_12*
	container *
shape:*
dtype0*(
_output_shapes
:
ß
Variable_12/Adadelta/AssignAssignVariable_12/Adadelta&Variable_12/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*(
_output_shapes
:

Variable_12/Adadelta/readIdentityVariable_12/Adadelta*
T0*
_class
loc:@Variable_12*(
_output_shapes
:
ą
8Variable_12/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_12*
dtype0*
_output_shapes
:

.Variable_12/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
˙
(Variable_12/Adadelta_1/Initializer/zerosFill8Variable_12/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_12/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_12*(
_output_shapes
:
ž
Variable_12/Adadelta_1
VariableV2*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_12*
	container *
shape:
ĺ
Variable_12/Adadelta_1/AssignAssignVariable_12/Adadelta_1(Variable_12/Adadelta_1/Initializer/zeros*(
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(

Variable_12/Adadelta_1/readIdentityVariable_12/Adadelta_1*
T0*
_class
loc:@Variable_12*(
_output_shapes
:
Ą
6Variable_13/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_13*
dtype0*
_output_shapes
:

,Variable_13/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
: 
ě
&Variable_13/Adadelta/Initializer/zerosFill6Variable_13/Adadelta/Initializer/zeros/shape_as_tensor,Variable_13/Adadelta/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*
_class
loc:@Variable_13
˘
Variable_13/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_13*
	container *
shape:
Ň
Variable_13/Adadelta/AssignAssignVariable_13/Adadelta&Variable_13/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:

Variable_13/Adadelta/readIdentityVariable_13/Adadelta*
T0*
_class
loc:@Variable_13*
_output_shapes	
:
Ł
8Variable_13/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_13*
dtype0*
_output_shapes
:

.Variable_13/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
: 
ň
(Variable_13/Adadelta_1/Initializer/zerosFill8Variable_13/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_13/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*
_class
loc:@Variable_13
¤
Variable_13/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_13*
	container *
shape:
Ř
Variable_13/Adadelta_1/AssignAssignVariable_13/Adadelta_1(Variable_13/Adadelta_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_13

Variable_13/Adadelta_1/readIdentityVariable_13/Adadelta_1*
T0*
_class
loc:@Variable_13*
_output_shapes	
:
Á
Fbatch_normalization_6/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_6/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_6/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_6/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_6/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
Â
$batch_normalization_6/gamma/Adadelta
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_6/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

+batch_normalization_6/gamma/Adadelta/AssignAssign$batch_normalization_6/gamma/Adadelta6batch_normalization_6/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_6/gamma/Adadelta/readIdentity$batch_normalization_6/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_6/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0
ł
>batch_normalization_6/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_6/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_6/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_6/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
Ä
&batch_normalization_6/gamma/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_6/gamma*
	container *
shape:

-batch_normalization_6/gamma/Adadelta_1/AssignAssign&batch_normalization_6/gamma/Adadelta_18batch_normalization_6/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_6/gamma/Adadelta_1/readIdentity&batch_normalization_6/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
ż
Ebatch_normalization_6/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_6/beta/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_6/beta*
dtype0
¨
5batch_normalization_6/beta/Adadelta/Initializer/zerosFillEbatch_normalization_6/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_6/beta/Adadelta/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*-
_class#
!loc:@batch_normalization_6/beta
Ŕ
#batch_normalization_6/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_6/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

*batch_normalization_6/beta/Adadelta/AssignAssign#batch_normalization_6/beta/Adadelta5batch_normalization_6/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_6/beta/Adadelta/readIdentity#batch_normalization_6/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
Á
Gbatch_normalization_6/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_6/beta/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_6/beta*
dtype0
Ž
7batch_normalization_6/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_6/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_6/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
Â
%batch_normalization_6/beta/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_6/beta*
	container 

,batch_normalization_6/beta/Adadelta_1/AssignAssign%batch_normalization_6/beta/Adadelta_17batch_normalization_6/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_6/beta/Adadelta_1/readIdentity%batch_normalization_6/beta/Adadelta_1*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_6/beta
Ż
6Variable_14/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_14*
dtype0*
_output_shapes
:

,Variable_14/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_14*
dtype0*
_output_shapes
: 
ů
&Variable_14/Adadelta/Initializer/zerosFill6Variable_14/Adadelta/Initializer/zeros/shape_as_tensor,Variable_14/Adadelta/Initializer/zeros/Const*(
_output_shapes
:*
T0*

index_type0*
_class
loc:@Variable_14
ź
Variable_14/Adadelta
VariableV2*
_class
loc:@Variable_14*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
ß
Variable_14/Adadelta/AssignAssignVariable_14/Adadelta&Variable_14/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*(
_output_shapes
:

Variable_14/Adadelta/readIdentityVariable_14/Adadelta*
_class
loc:@Variable_14*(
_output_shapes
:*
T0
ą
8Variable_14/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_14*
dtype0*
_output_shapes
:

.Variable_14/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_14*
dtype0*
_output_shapes
: 
˙
(Variable_14/Adadelta_1/Initializer/zerosFill8Variable_14/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_14/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_14*(
_output_shapes
:
ž
Variable_14/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_14*
	container *
shape:*
dtype0*(
_output_shapes
:
ĺ
Variable_14/Adadelta_1/AssignAssignVariable_14/Adadelta_1(Variable_14/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*(
_output_shapes
:

Variable_14/Adadelta_1/readIdentityVariable_14/Adadelta_1*
T0*
_class
loc:@Variable_14*(
_output_shapes
:
Ą
6Variable_15/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

,Variable_15/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
: 
ě
&Variable_15/Adadelta/Initializer/zerosFill6Variable_15/Adadelta/Initializer/zeros/shape_as_tensor,Variable_15/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_15*
_output_shapes	
:
˘
Variable_15/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:*
dtype0*
_output_shapes	
:
Ň
Variable_15/Adadelta/AssignAssignVariable_15/Adadelta&Variable_15/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:

Variable_15/Adadelta/readIdentityVariable_15/Adadelta*
T0*
_class
loc:@Variable_15*
_output_shapes	
:
Ł
8Variable_15/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

.Variable_15/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
: 
ň
(Variable_15/Adadelta_1/Initializer/zerosFill8Variable_15/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_15/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_15*
_output_shapes	
:
¤
Variable_15/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
Variable_15/Adadelta_1/AssignAssignVariable_15/Adadelta_1(Variable_15/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:

Variable_15/Adadelta_1/readIdentityVariable_15/Adadelta_1*
_output_shapes	
:*
T0*
_class
loc:@Variable_15
Á
Fbatch_normalization_7/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_7/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_7/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_7/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_7/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
Â
$batch_normalization_7/gamma/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_7/gamma*
	container *
shape:

+batch_normalization_7/gamma/Adadelta/AssignAssign$batch_normalization_7/gamma/Adadelta6batch_normalization_7/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_7/gamma/Adadelta/readIdentity$batch_normalization_7/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_7/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_7/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_7/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_7/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_7/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_7/gamma
Ä
&batch_normalization_7/gamma/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_7/gamma*
	container *
shape:

-batch_normalization_7/gamma/Adadelta_1/AssignAssign&batch_normalization_7/gamma/Adadelta_18batch_normalization_7/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_7/gamma/Adadelta_1/readIdentity&batch_normalization_7/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
ż
Ebatch_normalization_7/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_7/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_7/beta/Adadelta/Initializer/zerosFillEbatch_normalization_7/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_7/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
Ŕ
#batch_normalization_7/beta/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_7/beta*
	container *
shape:

*batch_normalization_7/beta/Adadelta/AssignAssign#batch_normalization_7/beta/Adadelta5batch_normalization_7/beta/Adadelta/Initializer/zeros*-
_class#
!loc:@batch_normalization_7/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ž
(batch_normalization_7/beta/Adadelta/readIdentity#batch_normalization_7/beta/Adadelta*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_7/beta
Á
Gbatch_normalization_7/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_7/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_7/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_7/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_7/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
Â
%batch_normalization_7/beta/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_7/beta*
	container 

,batch_normalization_7/beta/Adadelta_1/AssignAssign%batch_normalization_7/beta/Adadelta_17batch_normalization_7/beta/Adadelta_1/Initializer/zeros*-
_class#
!loc:@batch_normalization_7/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˛
*batch_normalization_7/beta/Adadelta_1/readIdentity%batch_normalization_7/beta/Adadelta_1*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_7/beta
Ż
6Variable_16/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"         @   *
_class
loc:@Variable_16*
dtype0*
_output_shapes
:

,Variable_16/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_16*
dtype0*
_output_shapes
: 
ř
&Variable_16/Adadelta/Initializer/zerosFill6Variable_16/Adadelta/Initializer/zeros/shape_as_tensor,Variable_16/Adadelta/Initializer/zeros/Const*'
_output_shapes
:@*
T0*

index_type0*
_class
loc:@Variable_16
ş
Variable_16/Adadelta
VariableV2*
shape:@*
dtype0*'
_output_shapes
:@*
shared_name *
_class
loc:@Variable_16*
	container 
Ţ
Variable_16/Adadelta/AssignAssignVariable_16/Adadelta&Variable_16/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*'
_output_shapes
:@

Variable_16/Adadelta/readIdentityVariable_16/Adadelta*
T0*
_class
loc:@Variable_16*'
_output_shapes
:@
ą
8Variable_16/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         @   *
_class
loc:@Variable_16*
dtype0*
_output_shapes
:

.Variable_16/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_16*
dtype0*
_output_shapes
: 
ţ
(Variable_16/Adadelta_1/Initializer/zerosFill8Variable_16/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_16/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_16*'
_output_shapes
:@
ź
Variable_16/Adadelta_1
VariableV2*
shape:@*
dtype0*'
_output_shapes
:@*
shared_name *
_class
loc:@Variable_16*
	container 
ä
Variable_16/Adadelta_1/AssignAssignVariable_16/Adadelta_1(Variable_16/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*'
_output_shapes
:@

Variable_16/Adadelta_1/readIdentityVariable_16/Adadelta_1*'
_output_shapes
:@*
T0*
_class
loc:@Variable_16
 
6Variable_17/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@Variable_17*
dtype0*
_output_shapes
:

,Variable_17/Adadelta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_17
ë
&Variable_17/Adadelta/Initializer/zerosFill6Variable_17/Adadelta/Initializer/zeros/shape_as_tensor,Variable_17/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_17*
_output_shapes
:@
 
Variable_17/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_17*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ń
Variable_17/Adadelta/AssignAssignVariable_17/Adadelta&Variable_17/Adadelta/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(

Variable_17/Adadelta/readIdentityVariable_17/Adadelta*
T0*
_class
loc:@Variable_17*
_output_shapes
:@
˘
8Variable_17/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:@*
_class
loc:@Variable_17

.Variable_17/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_17*
dtype0
ń
(Variable_17/Adadelta_1/Initializer/zerosFill8Variable_17/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_17/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_17*
_output_shapes
:@
˘
Variable_17/Adadelta_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Variable_17
×
Variable_17/Adadelta_1/AssignAssignVariable_17/Adadelta_1(Variable_17/Adadelta_1/Initializer/zeros*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:@*
use_locking(

Variable_17/Adadelta_1/readIdentityVariable_17/Adadelta_1*
T0*
_class
loc:@Variable_17*
_output_shapes
:@
Ŕ
Fbatch_normalization_8/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_8/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
: 
Ť
6batch_normalization_8/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_8/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_8/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
Ŕ
$batch_normalization_8/gamma/Adadelta
VariableV2*.
_class$
" loc:@batch_normalization_8/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 

+batch_normalization_8/gamma/Adadelta/AssignAssign$batch_normalization_8/gamma/Adadelta6batch_normalization_8/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
validate_shape(*
_output_shapes
:@
°
)batch_normalization_8/gamma/Adadelta/readIdentity$batch_normalization_8/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
Â
Hbatch_normalization_8/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_8/gamma/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_8/gamma
ą
8batch_normalization_8/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_8/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_8/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
Â
&batch_normalization_8/gamma/Adadelta_1
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_8/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@

-batch_normalization_8/gamma/Adadelta_1/AssignAssign&batch_normalization_8/gamma/Adadelta_18batch_normalization_8/gamma/Adadelta_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
validate_shape(
´
+batch_normalization_8/gamma/Adadelta_1/readIdentity&batch_normalization_8/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
ž
Ebatch_normalization_8/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_8/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
_output_shapes
: 
§
5batch_normalization_8/beta/Adadelta/Initializer/zerosFillEbatch_normalization_8/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_8/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
ž
#batch_normalization_8/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_8/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@

*batch_normalization_8/beta/Adadelta/AssignAssign#batch_normalization_8/beta/Adadelta5batch_normalization_8/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_8/beta*
validate_shape(*
_output_shapes
:@
­
(batch_normalization_8/beta/Adadelta/readIdentity#batch_normalization_8/beta/Adadelta*
_output_shapes
:@*
T0*-
_class#
!loc:@batch_normalization_8/beta
Ŕ
Gbatch_normalization_8/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:@*-
_class#
!loc:@batch_normalization_8/beta
ą
=batch_normalization_8/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
_output_shapes
: 
­
7batch_normalization_8/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_8/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_8/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
Ŕ
%batch_normalization_8/beta/Adadelta_1
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_8/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@

,batch_normalization_8/beta/Adadelta_1/AssignAssign%batch_normalization_8/beta/Adadelta_17batch_normalization_8/beta/Adadelta_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_8/beta
ą
*batch_normalization_8/beta/Adadelta_1/readIdentity%batch_normalization_8/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
Ż
6Variable_18/Adadelta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"      @       *
_class
loc:@Variable_18*
dtype0

,Variable_18/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_18*
dtype0*
_output_shapes
: 
÷
&Variable_18/Adadelta/Initializer/zerosFill6Variable_18/Adadelta/Initializer/zeros/shape_as_tensor,Variable_18/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
¸
Variable_18/Adadelta
VariableV2*
shape:@ *
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@Variable_18*
	container 
Ý
Variable_18/Adadelta/AssignAssignVariable_18/Adadelta&Variable_18/Adadelta/Initializer/zeros*&
_output_shapes
:@ *
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(

Variable_18/Adadelta/readIdentityVariable_18/Adadelta*
T0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
ą
8Variable_18/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @       *
_class
loc:@Variable_18

.Variable_18/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_18*
dtype0*
_output_shapes
: 
ý
(Variable_18/Adadelta_1/Initializer/zerosFill8Variable_18/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_18/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
ş
Variable_18/Adadelta_1
VariableV2*
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@Variable_18*
	container *
shape:@ 
ă
Variable_18/Adadelta_1/AssignAssignVariable_18/Adadelta_1(Variable_18/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:@ 

Variable_18/Adadelta_1/readIdentityVariable_18/Adadelta_1*
T0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
 
6Variable_19/Adadelta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB: *
_class
loc:@Variable_19*
dtype0

,Variable_19/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
: 
ë
&Variable_19/Adadelta/Initializer/zerosFill6Variable_19/Adadelta/Initializer/zeros/shape_as_tensor,Variable_19/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_19*
_output_shapes
: 
 
Variable_19/Adadelta
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_19*
	container *
shape: 
Ń
Variable_19/Adadelta/AssignAssignVariable_19/Adadelta&Variable_19/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
: 

Variable_19/Adadelta/readIdentityVariable_19/Adadelta*
T0*
_class
loc:@Variable_19*
_output_shapes
: 
˘
8Variable_19/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:

.Variable_19/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
: 
ń
(Variable_19/Adadelta_1/Initializer/zerosFill8Variable_19/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_19/Adadelta_1/Initializer/zeros/Const*
_output_shapes
: *
T0*

index_type0*
_class
loc:@Variable_19
˘
Variable_19/Adadelta_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_19*
	container 
×
Variable_19/Adadelta_1/AssignAssignVariable_19/Adadelta_1(Variable_19/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
: 

Variable_19/Adadelta_1/readIdentityVariable_19/Adadelta_1*
_output_shapes
: *
T0*
_class
loc:@Variable_19
Ŕ
Fbatch_normalization_9/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB: *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_9/gamma/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0
Ť
6batch_normalization_9/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_9/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_9/gamma/Adadelta/Initializer/zeros/Const*
_output_shapes
: *
T0*

index_type0*.
_class$
" loc:@batch_normalization_9/gamma
Ŕ
$batch_normalization_9/gamma/Adadelta
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_9/gamma*
	container *
shape: 

+batch_normalization_9/gamma/Adadelta/AssignAssign$batch_normalization_9/gamma/Adadelta6batch_normalization_9/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
validate_shape(*
_output_shapes
: 
°
)batch_normalization_9/gamma/Adadelta/readIdentity$batch_normalization_9/gamma/Adadelta*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: *
T0
Â
Hbatch_normalization_9/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB: *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_9/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
: 
ą
8batch_normalization_9/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_9/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_9/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
Â
&batch_normalization_9/gamma/Adadelta_1
VariableV2*.
_class$
" loc:@batch_normalization_9/gamma*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 

-batch_normalization_9/gamma/Adadelta_1/AssignAssign&batch_normalization_9/gamma/Adadelta_18batch_normalization_9/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
validate_shape(*
_output_shapes
: 
´
+batch_normalization_9/gamma/Adadelta_1/readIdentity&batch_normalization_9/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
ž
Ebatch_normalization_9/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB: *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_9/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
: 
§
5batch_normalization_9/beta/Adadelta/Initializer/zerosFillEbatch_normalization_9/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_9/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
ž
#batch_normalization_9/beta/Adadelta
VariableV2*-
_class#
!loc:@batch_normalization_9/beta*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 

*batch_normalization_9/beta/Adadelta/AssignAssign#batch_normalization_9/beta/Adadelta5batch_normalization_9/beta/Adadelta/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*-
_class#
!loc:@batch_normalization_9/beta*
validate_shape(
­
(batch_normalization_9/beta/Adadelta/readIdentity#batch_normalization_9/beta/Adadelta*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: *
T0
Ŕ
Gbatch_normalization_9/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB: *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_9/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
: 
­
7batch_normalization_9/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_9/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_9/beta/Adadelta_1/Initializer/zeros/Const*

index_type0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: *
T0
Ŕ
%batch_normalization_9/beta/Adadelta_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_9/beta*
	container *
shape: 

,batch_normalization_9/beta/Adadelta_1/AssignAssign%batch_normalization_9/beta/Adadelta_17batch_normalization_9/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_9/beta*
validate_shape(*
_output_shapes
: 
ą
*batch_normalization_9/beta/Adadelta_1/readIdentity%batch_normalization_9/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 

6Variable_20/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 

,Variable_20/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_20*
dtype0
ç
&Variable_20/Adadelta/Initializer/zerosFill6Variable_20/Adadelta/Initializer/zeros/shape_as_tensor,Variable_20/Adadelta/Initializer/zeros/Const*
_output_shapes
: *
T0*

index_type0*
_class
loc:@Variable_20

Variable_20/Adadelta
VariableV2*
_class
loc:@Variable_20*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Í
Variable_20/Adadelta/AssignAssignVariable_20/Adadelta&Variable_20/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*
_output_shapes
: 
|
Variable_20/Adadelta/readIdentityVariable_20/Adadelta*
_output_shapes
: *
T0*
_class
loc:@Variable_20

8Variable_20/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 

.Variable_20/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 
í
(Variable_20/Adadelta_1/Initializer/zerosFill8Variable_20/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_20/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_20*
_output_shapes
: 

Variable_20/Adadelta_1
VariableV2*
_class
loc:@Variable_20*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ó
Variable_20/Adadelta_1/AssignAssignVariable_20/Adadelta_1(Variable_20/Adadelta_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable_20

Variable_20/Adadelta_1/readIdentityVariable_20/Adadelta_1*
T0*
_class
loc:@Variable_20*
_output_shapes
: 

6Variable_21/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 

,Variable_21/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 
ç
&Variable_21/Adadelta/Initializer/zerosFill6Variable_21/Adadelta/Initializer/zeros/shape_as_tensor,Variable_21/Adadelta/Initializer/zeros/Const*
_output_shapes
: *
T0*

index_type0*
_class
loc:@Variable_21

Variable_21/Adadelta
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@Variable_21*
	container *
shape: *
dtype0
Í
Variable_21/Adadelta/AssignAssignVariable_21/Adadelta&Variable_21/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
: 
|
Variable_21/Adadelta/readIdentityVariable_21/Adadelta*
T0*
_class
loc:@Variable_21*
_output_shapes
: 

8Variable_21/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
: *
valueB *
_class
loc:@Variable_21*
dtype0

.Variable_21/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 
í
(Variable_21/Adadelta_1/Initializer/zerosFill8Variable_21/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_21/Adadelta_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_21*
_output_shapes
: *
T0

Variable_21/Adadelta_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable_21*
	container 
Ó
Variable_21/Adadelta_1/AssignAssignVariable_21/Adadelta_1(Variable_21/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
: 

Variable_21/Adadelta_1/readIdentityVariable_21/Adadelta_1*
_class
loc:@Variable_21*
_output_shapes
: *
T0
Ż
6Variable_22/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"             *
_class
loc:@Variable_22*
dtype0*
_output_shapes
:

,Variable_22/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_22*
dtype0
÷
&Variable_22/Adadelta/Initializer/zerosFill6Variable_22/Adadelta/Initializer/zeros/shape_as_tensor,Variable_22/Adadelta/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_22*&
_output_shapes
: *
T0
¸
Variable_22/Adadelta
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *
_class
loc:@Variable_22*
	container *
shape: 
Ý
Variable_22/Adadelta/AssignAssignVariable_22/Adadelta&Variable_22/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
: 

Variable_22/Adadelta/readIdentityVariable_22/Adadelta*
T0*
_class
loc:@Variable_22*&
_output_shapes
: 
ą
8Variable_22/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"             *
_class
loc:@Variable_22*
dtype0*
_output_shapes
:

.Variable_22/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_22*
dtype0*
_output_shapes
: 
ý
(Variable_22/Adadelta_1/Initializer/zerosFill8Variable_22/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_22/Adadelta_1/Initializer/zeros/Const*&
_output_shapes
: *
T0*

index_type0*
_class
loc:@Variable_22
ş
Variable_22/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_22*
	container *
shape: *
dtype0*&
_output_shapes
: 
ă
Variable_22/Adadelta_1/AssignAssignVariable_22/Adadelta_1(Variable_22/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
: 

Variable_22/Adadelta_1/readIdentityVariable_22/Adadelta_1*
T0*
_class
loc:@Variable_22*&
_output_shapes
: 
P
Adadelta/lrConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adadelta/rhoConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
U
Adadelta/epsilonConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
Š
&Adadelta/update_Variable/ApplyAdadeltaApplyAdadeltaVariableVariable/AdadeltaVariable/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable*&
_output_shapes
:@
¨
(Adadelta/update_Variable_1/ApplyAdadeltaApplyAdadelta
Variable_1Variable_1/AdadeltaVariable_1/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@Variable_1
Ó
7Adadelta/update_batch_normalization/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization/gamma"batch_normalization/gamma/Adadelta$batch_normalization/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_29*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
Î
6Adadelta/update_batch_normalization/beta/ApplyAdadeltaApplyAdadeltabatch_normalization/beta!batch_normalization/beta/Adadelta#batch_normalization/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_30*
_output_shapes
:@*
use_locking( *
T0*+
_class!
loc:@batch_normalization/beta
ś
(Adadelta/update_Variable_2/ApplyAdadeltaApplyAdadelta
Variable_2Variable_2/AdadeltaVariable_2/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*'
_output_shapes
:@
Ť
(Adadelta/update_Variable_3/ApplyAdadeltaApplyAdadelta
Variable_3Variable_3/AdadeltaVariable_3/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_3*
_output_shapes	
:
Ţ
9Adadelta/update_batch_normalization_1/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_1/gamma$batch_normalization_1/gamma/Adadelta&batch_normalization_1/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_26*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:
Ů
8Adadelta/update_batch_normalization_1/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_1/beta#batch_normalization_1/beta/Adadelta%batch_normalization_1/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_27*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
ˇ
(Adadelta/update_Variable_4/ApplyAdadeltaApplyAdadelta
Variable_4Variable_4/AdadeltaVariable_4/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*(
_output_shapes
:*
use_locking( *
T0*
_class
loc:@Variable_4
Ť
(Adadelta/update_Variable_5/ApplyAdadeltaApplyAdadelta
Variable_5Variable_5/AdadeltaVariable_5/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_5*
_output_shapes	
:*
use_locking( 
Ţ
9Adadelta/update_batch_normalization_2/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_2/gamma$batch_normalization_2/gamma/Adadelta&batch_normalization_2/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_23*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:
Ů
8Adadelta/update_batch_normalization_2/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_2/beta#batch_normalization_2/beta/Adadelta%batch_normalization_2/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_24*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
ˇ
(Adadelta/update_Variable_6/ApplyAdadeltaApplyAdadelta
Variable_6Variable_6/AdadeltaVariable_6/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*(
_output_shapes
:
Ť
(Adadelta/update_Variable_7/ApplyAdadeltaApplyAdadelta
Variable_7Variable_7/AdadeltaVariable_7/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_7*
_output_shapes	
:
Ţ
9Adadelta/update_batch_normalization_3/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_3/gamma$batch_normalization_3/gamma/Adadelta&batch_normalization_3/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_20*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
Ů
8Adadelta/update_batch_normalization_3/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_3/beta#batch_normalization_3/beta/Adadelta%batch_normalization_3/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_21*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
ˇ
(Adadelta/update_Variable_8/ApplyAdadeltaApplyAdadelta
Variable_8Variable_8/AdadeltaVariable_8/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_8*(
_output_shapes
:
Ť
(Adadelta/update_Variable_9/ApplyAdadeltaApplyAdadelta
Variable_9Variable_9/AdadeltaVariable_9/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@Variable_9
Ţ
9Adadelta/update_batch_normalization_4/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_4/gamma$batch_normalization_4/gamma/Adadelta&batch_normalization_4/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_17*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
Ů
8Adadelta/update_batch_normalization_4/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_4/beta#batch_normalization_4/beta/Adadelta%batch_normalization_4/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_18*
_output_shapes	
:*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_4/beta
ź
)Adadelta/update_Variable_10/ApplyAdadeltaApplyAdadeltaVariable_10Variable_10/AdadeltaVariable_10/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_5_grad/tuple/control_dependency_1*(
_output_shapes
:*
use_locking( *
T0*
_class
loc:@Variable_10
°
)Adadelta/update_Variable_11/ApplyAdadeltaApplyAdadeltaVariable_11Variable_11/AdadeltaVariable_11/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_5_grad/tuple/control_dependency_1*
_class
loc:@Variable_11*
_output_shapes	
:*
use_locking( *
T0
Ţ
9Adadelta/update_batch_normalization_5/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_5/gamma$batch_normalization_5/gamma/Adadelta&batch_normalization_5/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_14*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
Ů
8Adadelta/update_batch_normalization_5/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_5/beta#batch_normalization_5/beta/Adadelta%batch_normalization_5/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_15*
_output_shapes	
:*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_5/beta
ź
)Adadelta/update_Variable_12/ApplyAdadeltaApplyAdadeltaVariable_12Variable_12/AdadeltaVariable_12/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_6_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_12*(
_output_shapes
:
°
)Adadelta/update_Variable_13/ApplyAdadeltaApplyAdadeltaVariable_13Variable_13/AdadeltaVariable_13/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_6_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_13*
_output_shapes	
:
Ţ
9Adadelta/update_batch_normalization_6/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_6/gamma$batch_normalization_6/gamma/Adadelta&batch_normalization_6/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_11*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:*
use_locking( 
Ů
8Adadelta/update_batch_normalization_6/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_6/beta#batch_normalization_6/beta/Adadelta%batch_normalization_6/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_12*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
ź
)Adadelta/update_Variable_14/ApplyAdadeltaApplyAdadeltaVariable_14Variable_14/AdadeltaVariable_14/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_7_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_14*(
_output_shapes
:
°
)Adadelta/update_Variable_15/ApplyAdadeltaApplyAdadeltaVariable_15Variable_15/AdadeltaVariable_15/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_7_grad/tuple/control_dependency_1*
_class
loc:@Variable_15*
_output_shapes	
:*
use_locking( *
T0
Ý
9Adadelta/update_batch_normalization_7/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_7/gamma$batch_normalization_7/gamma/Adadelta&batch_normalization_7/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_8*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
Ř
8Adadelta/update_batch_normalization_7/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_7/beta#batch_normalization_7/beta/Adadelta%batch_normalization_7/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_9*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
ť
)Adadelta/update_Variable_16/ApplyAdadeltaApplyAdadeltaVariable_16Variable_16/AdadeltaVariable_16/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_16*'
_output_shapes
:@
Ż
)Adadelta/update_Variable_17/ApplyAdadeltaApplyAdadeltaVariable_17Variable_17/AdadeltaVariable_17/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_17*
_output_shapes
:@
Ü
9Adadelta/update_batch_normalization_8/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_8/gamma$batch_normalization_8/gamma/Adadelta&batch_normalization_8/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_5*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
×
8Adadelta/update_batch_normalization_8/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_8/beta#batch_normalization_8/beta/Adadelta%batch_normalization_8/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_6*
_output_shapes
:@*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_8/beta
ş
)Adadelta/update_Variable_18/ApplyAdadeltaApplyAdadeltaVariable_18Variable_18/AdadeltaVariable_18/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_9_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
Ż
)Adadelta/update_Variable_19/ApplyAdadeltaApplyAdadeltaVariable_19Variable_19/AdadeltaVariable_19/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_9_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_19*
_output_shapes
: 
Ü
9Adadelta/update_batch_normalization_9/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_9/gamma$batch_normalization_9/gamma/Adadelta&batch_normalization_9/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_2*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
×
8Adadelta/update_batch_normalization_9/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_9/beta#batch_normalization_9/beta/Adadelta%batch_normalization_9/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_3*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
Ľ
)Adadelta/update_Variable_20/ApplyAdadeltaApplyAdadeltaVariable_20Variable_20/AdadeltaVariable_20/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon-gradients/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_20*
_output_shapes
: 
Ľ
)Adadelta/update_Variable_21/ApplyAdadeltaApplyAdadeltaVariable_21Variable_21/AdadeltaVariable_21/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_21*
_output_shapes
: 
ť
)Adadelta/update_Variable_22/ApplyAdadeltaApplyAdadeltaVariable_22Variable_22/AdadeltaVariable_22/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/Conv2D_10_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_22*&
_output_shapes
: 

AdadeltaNoOp'^Adadelta/update_Variable/ApplyAdadelta)^Adadelta/update_Variable_1/ApplyAdadelta8^Adadelta/update_batch_normalization/gamma/ApplyAdadelta7^Adadelta/update_batch_normalization/beta/ApplyAdadelta)^Adadelta/update_Variable_2/ApplyAdadelta)^Adadelta/update_Variable_3/ApplyAdadelta:^Adadelta/update_batch_normalization_1/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_1/beta/ApplyAdadelta)^Adadelta/update_Variable_4/ApplyAdadelta)^Adadelta/update_Variable_5/ApplyAdadelta:^Adadelta/update_batch_normalization_2/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_2/beta/ApplyAdadelta)^Adadelta/update_Variable_6/ApplyAdadelta)^Adadelta/update_Variable_7/ApplyAdadelta:^Adadelta/update_batch_normalization_3/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_3/beta/ApplyAdadelta)^Adadelta/update_Variable_8/ApplyAdadelta)^Adadelta/update_Variable_9/ApplyAdadelta:^Adadelta/update_batch_normalization_4/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_4/beta/ApplyAdadelta*^Adadelta/update_Variable_10/ApplyAdadelta*^Adadelta/update_Variable_11/ApplyAdadelta:^Adadelta/update_batch_normalization_5/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_5/beta/ApplyAdadelta*^Adadelta/update_Variable_12/ApplyAdadelta*^Adadelta/update_Variable_13/ApplyAdadelta:^Adadelta/update_batch_normalization_6/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_6/beta/ApplyAdadelta*^Adadelta/update_Variable_14/ApplyAdadelta*^Adadelta/update_Variable_15/ApplyAdadelta:^Adadelta/update_batch_normalization_7/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_7/beta/ApplyAdadelta*^Adadelta/update_Variable_16/ApplyAdadelta*^Adadelta/update_Variable_17/ApplyAdadelta:^Adadelta/update_batch_normalization_8/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_8/beta/ApplyAdadelta*^Adadelta/update_Variable_18/ApplyAdadelta*^Adadelta/update_Variable_19/ApplyAdadelta:^Adadelta/update_batch_normalization_9/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_9/beta/ApplyAdadelta*^Adadelta/update_Variable_20/ApplyAdadelta*^Adadelta/update_Variable_21/ApplyAdadelta*^Adadelta/update_Variable_22/ApplyAdadelta
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: "}zňMX     L	~§ŻÖAJŔ°
Á!Ą!
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
ľ
ApplyAdadelta
var"T
accum"T
accum_update"T
lr"T
rho"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignSub
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ë
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	

FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%ˇŃ8"
data_formatstringNHWC"
is_trainingbool(
°
FusedBatchNormGrad

y_backprop"T
x"T

scale"T
reserve_space_1"T
reserve_space_2"T

x_backprop"T
scale_backprop"T
offset_backprop"T
reserve_space_3"T
reserve_space_4"T"
Ttype:
2"
epsilonfloat%ˇŃ8"
data_formatstringNHWC"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
x
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2		"
align_cornersbool( 
p
ResizeNearestNeighborGrad

grads"T
size
output"T"
Ttype:

2"
align_cornersbool( 
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
:
SqrtGrad
y"T
dy"T
z"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.6.02v1.6.0-0-gd2e24b6039Ů
P
PlaceholderPlaceholder*
dtype0
*
_output_shapes
:*
shape:
x
xPlaceholder*&
shape:˙˙˙˙˙˙˙˙˙*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
yPlaceholder*&
shape:˙˙˙˙˙˙˙˙˙*
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
truncated_normal/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0
˘
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*&
_output_shapes
:@*
seed2 *

seed *
T0

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:@
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*&
_output_shapes
:@*
T0

Variable
VariableV2*
shape:@*
shared_name *
dtype0*&
_output_shapes
:@*
	container 
Ź
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:@
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:@
R
ConstConst*
valueB@*ÍĚĚ=*
dtype0*
_output_shapes
:@
v

Variable_1
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 

Variable_1/AssignAssign
Variable_1Const*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:@
Ě
Conv2DConv2DxVariable/read*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
	dilations
*
T0*
strides
*
data_formatNHWC
˛
:batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:@*,
_class"
 loc:@batch_normalization/gamma
Ł
0batch_normalization/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
: 

*batch_normalization/gamma/Initializer/onesFill:batch_normalization/gamma/Initializer/ones/shape_as_tensor0batch_normalization/gamma/Initializer/ones/Const*

index_type0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
T0
ł
batch_normalization/gamma
VariableV2*,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
í
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@

batch_normalization/gamma/readIdentitybatch_normalization/gamma*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
ą
:batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*
valueB:@*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
˘
0batch_normalization/beta/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
: 

*batch_normalization/beta/Initializer/zerosFill:batch_normalization/beta/Initializer/zeros/shape_as_tensor0batch_normalization/beta/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ą
batch_normalization/beta
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container 
ę
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@

batch_normalization/beta/readIdentitybatch_normalization/beta*
_output_shapes
:@*
T0*+
_class!
loc:@batch_normalization/beta
ż
Abatch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:@*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
:
°
7batch_normalization/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
: 
 
1batch_normalization/moving_mean/Initializer/zerosFillAbatch_normalization/moving_mean/Initializer/zeros/shape_as_tensor7batch_normalization/moving_mean/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
ż
batch_normalization/moving_mean
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean

&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:@
Ş
$batch_normalization/moving_mean/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
Ć
Dbatch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:@*6
_class,
*(loc:@batch_normalization/moving_variance
ˇ
:batch_normalization/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
­
4batch_normalization/moving_variance/Initializer/onesFillDbatch_normalization/moving_variance/Initializer/ones/shape_as_tensor:batch_normalization/moving_variance/Initializer/ones/Const*

index_type0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
T0
Ç
#batch_normalization/moving_variance
VariableV2*
_output_shapes
:@*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:@*
dtype0

*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
ś
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
j
batch_normalization/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
s
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
_output_shapes
:*
T0

q
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
_output_shapes
:*
T0

\
 batch_normalization/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
Ľ
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:12batch_normalization/cond/FusedBatchNorm/Switch_2:1batch_normalization/cond/Const batch_normalization/cond/Const_1*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training(*
epsilon%o:
Ö
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchConv2D batch_normalization/cond/pred_id*
T0*
_class
loc:@Conv2D*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
Ő
0batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0*,
_class"
 loc:@batch_normalization/gamma
Ó
0batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0*+
_class!
loc:@batch_normalization/beta
Í
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_22batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( *
epsilon%o:*
T0
Ř
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchConv2D batch_normalization/cond/pred_id*
T0*
_class
loc:@Conv2D*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
×
2batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
:@:@
Ő
2batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
:@:@
ă
2batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read batch_normalization/cond/pred_id*
T0*2
_class(
&$loc:@batch_normalization/moving_mean* 
_output_shapes
:@:@
ë
2batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
Â
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: *
T0*
N
ą
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:@: 
ą
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
N*
_output_shapes

:@: *
T0
l
!batch_normalization/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization/cond_1/switch_tIdentity#batch_normalization/cond_1/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization/cond_1/switch_fIdentity!batch_normalization/cond_1/Switch*
_output_shapes
:*
T0

^
"batch_normalization/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization/cond_1/ConstConst$^batch_normalization/cond_1/switch_t*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0

"batch_normalization/cond_1/Const_1Const$^batch_normalization/cond_1/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0

 batch_normalization/cond_1/MergeMerge"batch_normalization/cond_1/Const_1 batch_normalization/cond_1/Const*
N*
_output_shapes
: : *
T0
Ž
(batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
Ó
'batch_normalization/AssignMovingAvg/SubSub(batch_normalization/AssignMovingAvg/read batch_normalization/cond/Merge_1*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
Ň
'batch_normalization/AssignMovingAvg/MulMul'batch_normalization/AssignMovingAvg/Sub batch_normalization/cond_1/Merge*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@*
T0
ć
#batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/Mul*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@*
use_locking( *
T0
¸
*batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
T0
Ű
)batch_normalization/AssignMovingAvg_1/SubSub*batch_normalization/AssignMovingAvg_1/read batch_normalization/cond/Merge_2*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
Ú
)batch_normalization/AssignMovingAvg_1/MulMul)batch_normalization/AssignMovingAvg_1/Sub batch_normalization/cond_1/Merge*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
ň
%batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/Mul*
_output_shapes
:@*
use_locking( *
T0*6
_class,
*(loc:@batch_normalization/moving_variance

BiasAddBiasAddbatch_normalization/cond/MergeVariable_1/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
T0
q
truncated_normal_1/shapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
§
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
T0*
dtype0*'
_output_shapes
:@*
seed2 

truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*'
_output_shapes
:@
|
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*'
_output_shapes
:@


Variable_2
VariableV2*'
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
ľ
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
_class
loc:@Variable_2*
validate_shape(*'
_output_shapes
:@*
use_locking(
x
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*'
_output_shapes
:@
V
Const_1Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_3
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

Variable_3/AssignAssign
Variable_3Const_1*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
l
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes	
:
×
Conv2D_1Conv2DBiasAddVariable_2/read*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ˇ
<batch_normalization_1/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_1/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_1/gamma/Initializer/onesFill<batch_normalization_1/gamma/Initializer/ones/shape_as_tensor2batch_normalization_1/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:
š
batch_normalization_1/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:
ö
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(

 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
T0
ś
<batch_normalization_1/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
Ś
2batch_normalization_1/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta

,batch_normalization_1/beta/Initializer/zerosFill<batch_normalization_1/beta/Initializer/zeros/shape_as_tensor2batch_normalization_1/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
ˇ
batch_normalization_1/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:
ó
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta

batch_normalization_1/beta/readIdentitybatch_normalization_1/beta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
Ä
Cbatch_normalization_1/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_1/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_1/moving_mean/Initializer/zerosFillCbatch_normalization_1/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_1/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
Ĺ
!batch_normalization_1/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:

(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_1/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_1/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_1/moving_variance/Initializer/onesFillFbatch_normalization_1/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_1/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
Í
%batch_normalization_1/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:

,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(
˝
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
l
!batch_normalization_1/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_1/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_1/cond/ConstConst$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_1/cond/Const_1Const$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
ś
)batch_normalization_1/cond/FusedBatchNormFusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm/Switch:14batch_normalization_1/cond/FusedBatchNorm/Switch_1:14batch_normalization_1/cond/FusedBatchNorm/Switch_2:1 batch_normalization_1/cond/Const"batch_normalization_1/cond/Const_1*
data_formatNHWC*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ::::*
is_training(*
epsilon%o:*
T0
ŕ
0batch_normalization_1/cond/FusedBatchNorm/SwitchSwitchConv2D_1"batch_normalization_1/cond/pred_id*
T0*
_class
loc:@Conv2D_1*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ
ß
2batch_normalization_1/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*"
_output_shapes
::
Ý
2batch_normalization_1/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_1/beta
Ţ
+batch_normalization_1/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm_1/Switch4batch_normalization_1/cond/FusedBatchNorm_1/Switch_14batch_normalization_1/cond/FusedBatchNorm_1/Switch_24batch_normalization_1/cond/FusedBatchNorm_1/Switch_34batch_normalization_1/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ::::*
is_training( *
epsilon%o:*
T0
â
2batch_normalization_1/cond/FusedBatchNorm_1/SwitchSwitchConv2D_1"batch_normalization_1/cond/pred_id*
T0*
_class
loc:@Conv2D_1*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ
á
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*"
_output_shapes
::
ß
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*"
_output_shapes
::
í
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read"batch_normalization_1/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*"
_output_shapes
::
ő
4batch_normalization_1/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read"batch_normalization_1/cond/pred_id*"
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
É
 batch_normalization_1/cond/MergeMerge+batch_normalization_1/cond/FusedBatchNorm_1)batch_normalization_1/cond/FusedBatchNorm*
T0*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙ČČ: 
¸
"batch_normalization_1/cond/Merge_1Merge-batch_normalization_1/cond/FusedBatchNorm_1:1+batch_normalization_1/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_1/cond/Merge_2Merge-batch_normalization_1/cond/FusedBatchNorm_1:2+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_1/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_1/cond_1/switch_tIdentity%batch_normalization_1/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_1/cond_1/switch_fIdentity#batch_normalization_1/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_1/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_1/cond_1/ConstConst&^batch_normalization_1/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_1/cond_1/Const_1Const&^batch_normalization_1/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_1/cond_1/MergeMerge$batch_normalization_1/cond_1/Const_1"batch_normalization_1/cond_1/Const*
N*
_output_shapes
: : *
T0
ľ
*batch_normalization_1/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_1/AssignMovingAvg/SubSub*batch_normalization_1/AssignMovingAvg/read"batch_normalization_1/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_1/AssignMovingAvg/MulMul)batch_normalization_1/AssignMovingAvg/Sub"batch_normalization_1/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
ď
%batch_normalization_1/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes	
:
ż
,batch_normalization_1/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
ä
+batch_normalization_1/AssignMovingAvg_1/SubSub,batch_normalization_1/AssignMovingAvg_1/read"batch_normalization_1/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
ă
+batch_normalization_1/AssignMovingAvg_1/MulMul+batch_normalization_1/AssignMovingAvg_1/Sub"batch_normalization_1/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:
ű
'batch_normalization_1/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes	
:

	BiasAdd_1BiasAdd batch_normalization_1/cond/MergeVariable_3/read*
T0*
data_formatNHWC*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ
q
truncated_normal_2/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
T0*
dtype0*(
_output_shapes
:*
seed2 

truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*(
_output_shapes
:*
T0
}
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*(
_output_shapes
:


Variable_4
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ś
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*(
_output_shapes
:
y
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*(
_output_shapes
:
V
Const_2Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_5
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 

Variable_5/AssignAssign
Variable_5Const_2*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
l
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes	
:
×
Conv2D_2Conv2D	BiasAdd_1Variable_4/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ˇ
<batch_normalization_2/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_2/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_2/gamma/Initializer/onesFill<batch_normalization_2/gamma/Initializer/ones/shape_as_tensor2batch_normalization_2/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:
š
batch_normalization_2/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ö
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes	
:

 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:
ś
<batch_normalization_2/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_2/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 

,batch_normalization_2/beta/Initializer/zerosFill<batch_normalization_2/beta/Initializer/zeros/shape_as_tensor2batch_normalization_2/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
ˇ
batch_normalization_2/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container 
ó
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_2/beta/readIdentitybatch_normalization_2/beta*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_2/beta
Ä
Cbatch_normalization_2/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_2/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_2/moving_mean/Initializer/zerosFillCbatch_normalization_2/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_2/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
Ĺ
!batch_normalization_2/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_2/moving_mean

(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ą
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_2/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_2/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_2/moving_variance/Initializer/onesFillFbatch_normalization_2/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_2/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
Í
%batch_normalization_2/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:

,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
˝
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
l
!batch_normalization_2/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
_output_shapes
:*
T0

^
"batch_normalization_2/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:14batch_normalization_2/cond/FusedBatchNorm/Switch_2:1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training(*
epsilon%o:
Ü
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchConv2D_2"batch_normalization_2/cond/pred_id*
T0*
_class
loc:@Conv2D_2*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
ß
2batch_normalization_2/cond/FusedBatchNorm/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*.
_class$
" loc:@batch_normalization_2/gamma*"
_output_shapes
::*
T0
Ý
2batch_normalization_2/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*"
_output_shapes
::
Ü
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_24batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( 
Ţ
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchConv2D_2"batch_normalization_2/cond/pred_id*
T0*
_class
loc:@Conv2D_2*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
á
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*"
_output_shapes
::
ß
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*-
_class#
!loc:@batch_normalization_2/beta*"
_output_shapes
::*
T0
í
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*"
_output_shapes
::
ő
4batch_normalization_2/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*"
_output_shapes
::
Ç
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 
¸
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_2/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_2/cond_1/switch_tIdentity%batch_normalization_2/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_2/cond_1/switch_fIdentity#batch_normalization_2/cond_1/Switch*
_output_shapes
:*
T0

`
$batch_normalization_2/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_2/cond_1/ConstConst&^batch_normalization_2/cond_1/switch_t*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0

$batch_normalization_2/cond_1/Const_1Const&^batch_normalization_2/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_2/cond_1/MergeMerge$batch_normalization_2/cond_1/Const_1"batch_normalization_2/cond_1/Const*
N*
_output_shapes
: : *
T0
ľ
*batch_normalization_2/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
Ü
)batch_normalization_2/AssignMovingAvg/SubSub*batch_normalization_2/AssignMovingAvg/read"batch_normalization_2/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_2/AssignMovingAvg/MulMul)batch_normalization_2/AssignMovingAvg/Sub"batch_normalization_2/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes	
:
ď
%batch_normalization_2/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/Mul*
_output_shapes	
:*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
ż
,batch_normalization_2/AssignMovingAvg_1/readIdentity%batch_normalization_2/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
ä
+batch_normalization_2/AssignMovingAvg_1/SubSub,batch_normalization_2/AssignMovingAvg_1/read"batch_normalization_2/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
ă
+batch_normalization_2/AssignMovingAvg_1/MulMul+batch_normalization_2/AssignMovingAvg_1/Sub"batch_normalization_2/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:
ű
'batch_normalization_2/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes	
:

	BiasAdd_2BiasAdd batch_normalization_2/cond/MergeVariable_5/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
q
truncated_normal_3/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*(
_output_shapes
:*
seed2 *

seed *
T0

truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*(
_output_shapes
:
}
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*(
_output_shapes
:


Variable_6
VariableV2*
dtype0*(
_output_shapes
:*
	container *
shape:*
shared_name 
ś
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*(
_output_shapes
:
y
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*(
_output_shapes
:
V
Const_3Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_7
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

Variable_7/AssignAssign
Variable_7Const_3*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:
l
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
×
Conv2D_3Conv2D	BiasAdd_2Variable_6/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
ˇ
<batch_normalization_3/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_3/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_3/gamma/Initializer/onesFill<batch_normalization_3/gamma/Initializer/ones/shape_as_tensor2batch_normalization_3/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
š
batch_normalization_3/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ö
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma

 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
ś
<batch_normalization_3/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_3/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 

,batch_normalization_3/beta/Initializer/zerosFill<batch_normalization_3/beta/Initializer/zeros/shape_as_tensor2batch_normalization_3/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
ˇ
batch_normalization_3/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ó
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_3/beta/readIdentitybatch_normalization_3/beta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
Ä
Cbatch_normalization_3/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*4
_class*
(&loc:@batch_normalization_3/moving_mean
´
9batch_normalization_3/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
Š
3batch_normalization_3/moving_mean/Initializer/zerosFillCbatch_normalization_3/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_3/moving_mean/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_3/moving_mean
Ĺ
!batch_normalization_3/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:

(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_3/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_3/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_3/moving_variance/Initializer/onesFillFbatch_normalization_3/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_3/moving_variance/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_3/moving_variance
Í
%batch_normalization_3/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:

,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:
l
!batch_normalization_3/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

w
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_3/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:14batch_normalization_3/cond/FusedBatchNorm/Switch_2:1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(*
epsilon%o:*
T0*
data_formatNHWC
Ü
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchConv2D_3"batch_normalization_3/cond/pred_id*
T0*
_class
loc:@Conv2D_3*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
ß
2batch_normalization_3/cond/FusedBatchNorm/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*"
_output_shapes
::
Ý
2batch_normalization_3/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*"
_output_shapes
::
Ü
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_24batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( 
Ţ
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchConv2D_3"batch_normalization_3/cond/pred_id*
T0*
_class
loc:@Conv2D_3*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
á
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_3/gamma
ß
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*"
_output_shapes
::
í
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_3/moving_mean/read"batch_normalization_3/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*"
_output_shapes
::
ő
4batch_normalization_3/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_3/moving_variance/read"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*"
_output_shapes
::
Ç
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 
¸
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_3/cond_1/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

{
%batch_normalization_3/cond_1/switch_tIdentity%batch_normalization_3/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_3/cond_1/switch_fIdentity#batch_normalization_3/cond_1/Switch*
_output_shapes
:*
T0

`
$batch_normalization_3/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_3/cond_1/ConstConst&^batch_normalization_3/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_3/cond_1/Const_1Const&^batch_normalization_3/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_3/cond_1/MergeMerge$batch_normalization_3/cond_1/Const_1"batch_normalization_3/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_3/AssignMovingAvg/readIdentity!batch_normalization_3/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_3/AssignMovingAvg/SubSub*batch_normalization_3/AssignMovingAvg/read"batch_normalization_3/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_3/AssignMovingAvg/MulMul)batch_normalization_3/AssignMovingAvg/Sub"batch_normalization_3/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
ď
%batch_normalization_3/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:
ż
,batch_normalization_3/AssignMovingAvg_1/readIdentity%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:*
T0
ä
+batch_normalization_3/AssignMovingAvg_1/SubSub,batch_normalization_3/AssignMovingAvg_1/read"batch_normalization_3/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:
ă
+batch_normalization_3/AssignMovingAvg_1/MulMul+batch_normalization_3/AssignMovingAvg_1/Sub"batch_normalization_3/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:
ű
'batch_normalization_3/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:

	BiasAdd_3BiasAdd batch_normalization_3/cond/MergeVariable_7/read*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0
q
truncated_normal_4/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_4/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
T0*
dtype0*(
_output_shapes
:*
seed2 *

seed 

truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*(
_output_shapes
:*
T0
}
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*(
_output_shapes
:


Variable_8
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
ś
Variable_8/AssignAssign
Variable_8truncated_normal_4*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:
y
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*(
_output_shapes
:
V
Const_4Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
x

Variable_9
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:

Variable_9/AssignAssign
Variable_9Const_4*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:*
use_locking(
l
Variable_9/readIdentity
Variable_9*
_class
loc:@Variable_9*
_output_shapes	
:*
T0
×
Conv2D_4Conv2D	BiasAdd_3Variable_8/read*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ˇ
<batch_normalization_4/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_4/gamma/Initializer/ones/ConstConst*
_output_shapes
: *
valueB
 *  ?*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0

,batch_normalization_4/gamma/Initializer/onesFill<batch_normalization_4/gamma/Initializer/ones/shape_as_tensor2batch_normalization_4/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma
š
batch_normalization_4/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_4/gamma*
	container 
ö
"batch_normalization_4/gamma/AssignAssignbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:

 batch_normalization_4/gamma/readIdentitybatch_normalization_4/gamma*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
ś
<batch_normalization_4/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_4/beta
Ś
2batch_normalization_4/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 

,batch_normalization_4/beta/Initializer/zerosFill<batch_normalization_4/beta/Initializer/zeros/shape_as_tensor2batch_normalization_4/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
ˇ
batch_normalization_4/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_4/beta
ó
!batch_normalization_4/beta/AssignAssignbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_4/beta/readIdentitybatch_normalization_4/beta*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
Ä
Cbatch_normalization_4/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
´
9batch_normalization_4/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_4/moving_mean/Initializer/zerosFillCbatch_normalization_4/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_4/moving_mean/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_4/moving_mean
Ĺ
!batch_normalization_4/moving_mean
VariableV2*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container *
shape:*
dtype0

(batch_normalization_4/moving_mean/AssignAssign!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_4/moving_mean/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_4/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*8
_class.
,*loc:@batch_normalization_4/moving_variance
ť
<batch_normalization_4/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_4/moving_variance/Initializer/onesFillFbatch_normalization_4/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_4/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
Í
%batch_normalization_4/moving_variance
VariableV2*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container *
shape:*
dtype0

,batch_normalization_4/moving_variance/AssignAssign%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_4/moving_variance/readIdentity%batch_normalization_4/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
l
!batch_normalization_4/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

w
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_4/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 

"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
´
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:14batch_normalization_4/cond/FusedBatchNorm/Switch_2:1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(*
epsilon%o:
Ü
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchConv2D_4"batch_normalization_4/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0*
_class
loc:@Conv2D_4
ß
2batch_normalization_4/cond/FusedBatchNorm/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*"
_output_shapes
::
Ý
2batch_normalization_4/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*-
_class#
!loc:@batch_normalization_4/beta*"
_output_shapes
::*
T0
Ü
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_24batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:
Ţ
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchConv2D_4"batch_normalization_4/cond/pred_id*
T0*
_class
loc:@Conv2D_4*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
á
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*"
_output_shapes
::
ß
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_4/beta
í
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_4/moving_mean/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
ő
4batch_normalization_4/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_4/moving_variance/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
Ç
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 
¸
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:: 
n
#batch_normalization_4/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_4/cond_1/switch_tIdentity%batch_normalization_4/cond_1/Switch:1*
_output_shapes
:*
T0

y
%batch_normalization_4/cond_1/switch_fIdentity#batch_normalization_4/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_4/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_4/cond_1/ConstConst&^batch_normalization_4/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_4/cond_1/Const_1Const&^batch_normalization_4/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_4/cond_1/MergeMerge$batch_normalization_4/cond_1/Const_1"batch_normalization_4/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_4/AssignMovingAvg/readIdentity!batch_normalization_4/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_4/AssignMovingAvg/SubSub*batch_normalization_4/AssignMovingAvg/read"batch_normalization_4/cond/Merge_1*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
Ű
)batch_normalization_4/AssignMovingAvg/MulMul)batch_normalization_4/AssignMovingAvg/Sub"batch_normalization_4/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:
ď
%batch_normalization_4/AssignMovingAvg	AssignSub!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes	
:
ż
,batch_normalization_4/AssignMovingAvg_1/readIdentity%batch_normalization_4/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
ä
+batch_normalization_4/AssignMovingAvg_1/SubSub,batch_normalization_4/AssignMovingAvg_1/read"batch_normalization_4/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
ă
+batch_normalization_4/AssignMovingAvg_1/MulMul+batch_normalization_4/AssignMovingAvg_1/Sub"batch_normalization_4/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:
ű
'batch_normalization_4/AssignMovingAvg_1	AssignSub%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/Mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes	
:*
use_locking( *
T0

	BiasAdd_4BiasAdd batch_normalization_4/cond/MergeVariable_9/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
q
truncated_normal_5/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_5/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*

seed *
T0*
dtype0*(
_output_shapes
:*
seed2 

truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*(
_output_shapes
:*
T0
}
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*(
_output_shapes
:*
T0

Variable_10
VariableV2*
shape:*
shared_name *
dtype0*(
_output_shapes
:*
	container 
š
Variable_10/AssignAssignVariable_10truncated_normal_5*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*(
_output_shapes
:
|
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*(
_output_shapes
:
V
Const_5Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
y
Variable_11
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ą
Variable_11/AssignAssignVariable_11Const_5*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(
o
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
Ř
Conv2D_5Conv2D	BiasAdd_4Variable_10/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
ˇ
<batch_normalization_5/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_5/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_5/gamma/Initializer/onesFill<batch_normalization_5/gamma/Initializer/ones/shape_as_tensor2batch_normalization_5/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
š
batch_normalization_5/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma*
	container *
shape:
ö
"batch_normalization_5/gamma/AssignAssignbatch_normalization_5/gamma,batch_normalization_5/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

 batch_normalization_5/gamma/readIdentitybatch_normalization_5/gamma*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
ś
<batch_normalization_5/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_5/beta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta*
dtype0

,batch_normalization_5/beta/Initializer/zerosFill<batch_normalization_5/beta/Initializer/zeros/shape_as_tensor2batch_normalization_5/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
ˇ
batch_normalization_5/beta
VariableV2*-
_class#
!loc:@batch_normalization_5/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ó
!batch_normalization_5/beta/AssignAssignbatch_normalization_5/beta,batch_normalization_5/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_5/beta/readIdentitybatch_normalization_5/beta*
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Ä
Cbatch_normalization_5/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_5/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_5/moving_mean/Initializer/zerosFillCbatch_normalization_5/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_5/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
Ĺ
!batch_normalization_5/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_5/moving_mean*
	container *
shape:

(batch_normalization_5/moving_mean/AssignAssign!batch_normalization_5/moving_mean3batch_normalization_5/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_5/moving_mean/readIdentity!batch_normalization_5/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_5/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_5/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_5/moving_variance/Initializer/onesFillFbatch_normalization_5/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_5/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:
Í
%batch_normalization_5/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_5/moving_variance*
	container 

,batch_normalization_5/moving_variance/AssignAssign%batch_normalization_5/moving_variance6batch_normalization_5/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_5/moving_variance/readIdentity%batch_normalization_5/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:
l
!batch_normalization_5/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

w
#batch_normalization_5/cond/switch_tIdentity#batch_normalization_5/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_5/cond/switch_fIdentity!batch_normalization_5/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_5/cond/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


 batch_normalization_5/cond/ConstConst$^batch_normalization_5/cond/switch_t*
_output_shapes
: *
valueB *
dtype0

"batch_normalization_5/cond/Const_1Const$^batch_normalization_5/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_5/cond/FusedBatchNormFusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:14batch_normalization_5/cond/FusedBatchNorm/Switch_2:1 batch_normalization_5/cond/Const"batch_normalization_5/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(
Ü
0batch_normalization_5/cond/FusedBatchNorm/SwitchSwitchConv2D_5"batch_normalization_5/cond/pred_id*
T0*
_class
loc:@Conv2D_5*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
ß
2batch_normalization_5/cond/FusedBatchNorm/Switch_1Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_5/gamma
Ý
2batch_normalization_5/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*-
_class#
!loc:@batch_normalization_5/beta*"
_output_shapes
::*
T0
Ü
+batch_normalization_5/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_24batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( 
Ţ
2batch_normalization_5/cond/FusedBatchNorm_1/SwitchSwitchConv2D_5"batch_normalization_5/cond/pred_id*
T0*
_class
loc:@Conv2D_5*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
á
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_5/gamma
ß
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_5/beta*"
_output_shapes
::
í
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_5/moving_mean/read"batch_normalization_5/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*"
_output_shapes
::
ő
4batch_normalization_5/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_5/moving_variance/read"batch_normalization_5/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*"
_output_shapes
::
Ç
 batch_normalization_5/cond/MergeMerge+batch_normalization_5/cond/FusedBatchNorm_1)batch_normalization_5/cond/FusedBatchNorm*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: *
T0
¸
"batch_normalization_5/cond/Merge_1Merge-batch_normalization_5/cond/FusedBatchNorm_1:1+batch_normalization_5/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_5/cond/Merge_2Merge-batch_normalization_5/cond/FusedBatchNorm_1:2+batch_normalization_5/cond/FusedBatchNorm:2*
N*
_output_shapes
	:: *
T0
n
#batch_normalization_5/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_5/cond_1/switch_tIdentity%batch_normalization_5/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_5/cond_1/switch_fIdentity#batch_normalization_5/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_5/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_5/cond_1/ConstConst&^batch_normalization_5/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_5/cond_1/Const_1Const&^batch_normalization_5/cond_1/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ą
"batch_normalization_5/cond_1/MergeMerge$batch_normalization_5/cond_1/Const_1"batch_normalization_5/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_5/AssignMovingAvg/readIdentity!batch_normalization_5/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_5/AssignMovingAvg/SubSub*batch_normalization_5/AssignMovingAvg/read"batch_normalization_5/cond/Merge_1*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
Ű
)batch_normalization_5/AssignMovingAvg/MulMul)batch_normalization_5/AssignMovingAvg/Sub"batch_normalization_5/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
ď
%batch_normalization_5/AssignMovingAvg	AssignSub!batch_normalization_5/moving_mean)batch_normalization_5/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes	
:
ż
,batch_normalization_5/AssignMovingAvg_1/readIdentity%batch_normalization_5/moving_variance*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
ä
+batch_normalization_5/AssignMovingAvg_1/SubSub,batch_normalization_5/AssignMovingAvg_1/read"batch_normalization_5/cond/Merge_2*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
ă
+batch_normalization_5/AssignMovingAvg_1/MulMul+batch_normalization_5/AssignMovingAvg_1/Sub"batch_normalization_5/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:
ű
'batch_normalization_5/AssignMovingAvg_1	AssignSub%batch_normalization_5/moving_variance+batch_normalization_5/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes	
:

	BiasAdd_5BiasAdd batch_normalization_5/cond/MergeVariable_11/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
q
truncated_normal_6/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_6/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
T0*
dtype0*(
_output_shapes
:*
seed2 *

seed 

truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*(
_output_shapes
:
}
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0*(
_output_shapes
:

Variable_12
VariableV2*(
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
š
Variable_12/AssignAssignVariable_12truncated_normal_6*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*(
_output_shapes
:
|
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12*(
_output_shapes
:
V
Const_6Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes	
:
y
Variable_13
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ą
Variable_13/AssignAssignVariable_13Const_6*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:
o
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13*
_output_shapes	
:
Ř
Conv2D_6Conv2D	BiasAdd_5Variable_12/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
	dilations
*
T0
ˇ
<batch_normalization_6/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_6/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_6/gamma/Initializer/onesFill<batch_normalization_6/gamma/Initializer/ones/shape_as_tensor2batch_normalization_6/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
š
batch_normalization_6/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_6/gamma*
	container 
ö
"batch_normalization_6/gamma/AssignAssignbatch_normalization_6/gamma,batch_normalization_6/gamma/Initializer/ones*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

 batch_normalization_6/gamma/readIdentitybatch_normalization_6/gamma*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_6/gamma
ś
<batch_normalization_6/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_6/beta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
: 

,batch_normalization_6/beta/Initializer/zerosFill<batch_normalization_6/beta/Initializer/zeros/shape_as_tensor2batch_normalization_6/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
ˇ
batch_normalization_6/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_6/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ó
!batch_normalization_6/beta/AssignAssignbatch_normalization_6/beta,batch_normalization_6/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:

batch_normalization_6/beta/readIdentitybatch_normalization_6/beta*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:*
T0
Ä
Cbatch_normalization_6/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:*4
_class*
(&loc:@batch_normalization_6/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_6/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_6/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_6/moving_mean/Initializer/zerosFillCbatch_normalization_6/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_6/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
Ĺ
!batch_normalization_6/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *4
_class*
(&loc:@batch_normalization_6/moving_mean*
	container *
shape:

(batch_normalization_6/moving_mean/AssignAssign!batch_normalization_6/moving_mean3batch_normalization_6/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_6/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ą
&batch_normalization_6/moving_mean/readIdentity!batch_normalization_6/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_6/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_6/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_6/moving_variance/Initializer/onesFillFbatch_normalization_6/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_6/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:
Í
%batch_normalization_6/moving_variance
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization_6/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:

,batch_normalization_6/moving_variance/AssignAssign%batch_normalization_6/moving_variance6batch_normalization_6/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_6/moving_variance/readIdentity%batch_normalization_6/moving_variance*
_output_shapes	
:*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance
l
!batch_normalization_6/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

w
#batch_normalization_6/cond/switch_tIdentity#batch_normalization_6/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_6/cond/switch_fIdentity!batch_normalization_6/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_6/cond/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


 batch_normalization_6/cond/ConstConst$^batch_normalization_6/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_6/cond/Const_1Const$^batch_normalization_6/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_6/cond/FusedBatchNormFusedBatchNorm2batch_normalization_6/cond/FusedBatchNorm/Switch:14batch_normalization_6/cond/FusedBatchNorm/Switch_1:14batch_normalization_6/cond/FusedBatchNorm/Switch_2:1 batch_normalization_6/cond/Const"batch_normalization_6/cond/Const_1*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training(*
epsilon%o:
Ü
0batch_normalization_6/cond/FusedBatchNorm/SwitchSwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*
_class
loc:@Conv2D_6*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
ß
2batch_normalization_6/cond/FusedBatchNorm/Switch_1Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_6/gamma*"
_output_shapes
::
Ý
2batch_normalization_6/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_6/beta*"
_output_shapes
::
Ü
+batch_normalization_6/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_6/cond/FusedBatchNorm_1/Switch4batch_normalization_6/cond/FusedBatchNorm_1/Switch_14batch_normalization_6/cond/FusedBatchNorm_1/Switch_24batch_normalization_6/cond/FusedBatchNorm_1/Switch_34batch_normalization_6/cond/FusedBatchNorm_1/Switch_4*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:*
T0
Ţ
2batch_normalization_6/cond/FusedBatchNorm_1/SwitchSwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*
_class
loc:@Conv2D_6*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
á
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*.
_class$
" loc:@batch_normalization_6/gamma*"
_output_shapes
::*
T0
ß
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_6/beta*"
_output_shapes
::
í
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_6/moving_mean/read"batch_normalization_6/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*"
_output_shapes
::
ő
4batch_normalization_6/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_6/moving_variance/read"batch_normalization_6/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*"
_output_shapes
::
Ç
 batch_normalization_6/cond/MergeMerge+batch_normalization_6/cond/FusedBatchNorm_1)batch_normalization_6/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 
¸
"batch_normalization_6/cond/Merge_1Merge-batch_normalization_6/cond/FusedBatchNorm_1:1+batch_normalization_6/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_6/cond/Merge_2Merge-batch_normalization_6/cond/FusedBatchNorm_1:2+batch_normalization_6/cond/FusedBatchNorm:2*
_output_shapes
	:: *
T0*
N
n
#batch_normalization_6/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_6/cond_1/switch_tIdentity%batch_normalization_6/cond_1/Switch:1*
_output_shapes
:*
T0

y
%batch_normalization_6/cond_1/switch_fIdentity#batch_normalization_6/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_6/cond_1/pred_idIdentityPlaceholder*
_output_shapes
:*
T0


"batch_normalization_6/cond_1/ConstConst&^batch_normalization_6/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_6/cond_1/Const_1Const&^batch_normalization_6/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_6/cond_1/MergeMerge$batch_normalization_6/cond_1/Const_1"batch_normalization_6/cond_1/Const*
N*
_output_shapes
: : *
T0
ľ
*batch_normalization_6/AssignMovingAvg/readIdentity!batch_normalization_6/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_6/AssignMovingAvg/SubSub*batch_normalization_6/AssignMovingAvg/read"batch_normalization_6/cond/Merge_1*
_output_shapes	
:*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean
Ű
)batch_normalization_6/AssignMovingAvg/MulMul)batch_normalization_6/AssignMovingAvg/Sub"batch_normalization_6/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:
ď
%batch_normalization_6/AssignMovingAvg	AssignSub!batch_normalization_6/moving_mean)batch_normalization_6/AssignMovingAvg/Mul*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes	
:*
use_locking( 
ż
,batch_normalization_6/AssignMovingAvg_1/readIdentity%batch_normalization_6/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:
ä
+batch_normalization_6/AssignMovingAvg_1/SubSub,batch_normalization_6/AssignMovingAvg_1/read"batch_normalization_6/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:
ă
+batch_normalization_6/AssignMovingAvg_1/MulMul+batch_normalization_6/AssignMovingAvg_1/Sub"batch_normalization_6/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:
ű
'batch_normalization_6/AssignMovingAvg_1	AssignSub%batch_normalization_6/moving_variance+batch_normalization_6/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes	
:

	BiasAdd_6BiasAdd batch_normalization_6/cond/MergeVariable_13/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
k
ResizeNearestNeighbor/sizeConst*
valueB"d   d   *
dtype0*
_output_shapes
:
Ľ
ResizeNearestNeighborResizeNearestNeighbor	BiasAdd_6ResizeNearestNeighbor/size*
align_corners( *
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
q
truncated_normal_7/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_7/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
¨
"truncated_normal_7/TruncatedNormalTruncatedNormaltruncated_normal_7/shape*
dtype0*(
_output_shapes
:*
seed2 *

seed *
T0

truncated_normal_7/mulMul"truncated_normal_7/TruncatedNormaltruncated_normal_7/stddev*(
_output_shapes
:*
T0
}
truncated_normal_7Addtruncated_normal_7/multruncated_normal_7/mean*
T0*(
_output_shapes
:

Variable_14
VariableV2*
shared_name *
dtype0*(
_output_shapes
:*
	container *
shape:
š
Variable_14/AssignAssignVariable_14truncated_normal_7*
T0*
_class
loc:@Variable_14*
validate_shape(*(
_output_shapes
:*
use_locking(
|
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14*(
_output_shapes
:
V
Const_7Const*
_output_shapes	
:*
valueB*ÍĚĚ=*
dtype0
y
Variable_15
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ą
Variable_15/AssignAssignVariable_15Const_7*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(
o
Variable_15/readIdentityVariable_15*
_output_shapes	
:*
T0*
_class
loc:@Variable_15
ä
Conv2D_7Conv2DResizeNearestNeighborVariable_14/read*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ˇ
<batch_normalization_7/gamma/Initializer/ones/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_7/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_7/gamma/Initializer/onesFill<batch_normalization_7/gamma/Initializer/ones/shape_as_tensor2batch_normalization_7/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_7/gamma
š
batch_normalization_7/gamma
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_7/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ö
"batch_normalization_7/gamma/AssignAssignbatch_normalization_7/gamma,batch_normalization_7/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
validate_shape(*
_output_shapes	
:

 batch_normalization_7/gamma/readIdentitybatch_normalization_7/gamma*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
ś
<batch_normalization_7/beta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_7/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_7/beta

,batch_normalization_7/beta/Initializer/zerosFill<batch_normalization_7/beta/Initializer/zeros/shape_as_tensor2batch_normalization_7/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
ˇ
batch_normalization_7/beta
VariableV2*-
_class#
!loc:@batch_normalization_7/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ó
!batch_normalization_7/beta/AssignAssignbatch_normalization_7/beta,batch_normalization_7/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_7/beta

batch_normalization_7/beta/readIdentitybatch_normalization_7/beta*
T0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
Ä
Cbatch_normalization_7/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*4
_class*
(&loc:@batch_normalization_7/moving_mean*
dtype0
´
9batch_normalization_7/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_7/moving_mean*
dtype0*
_output_shapes
: 
Š
3batch_normalization_7/moving_mean/Initializer/zerosFillCbatch_normalization_7/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_7/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
Ĺ
!batch_normalization_7/moving_mean
VariableV2*
shared_name *4
_class*
(&loc:@batch_normalization_7/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:

(batch_normalization_7/moving_mean/AssignAssign!batch_normalization_7/moving_mean3batch_normalization_7/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
validate_shape(*
_output_shapes	
:
ą
&batch_normalization_7/moving_mean/readIdentity!batch_normalization_7/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
Ë
Fbatch_normalization_7/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:*8
_class.
,*loc:@batch_normalization_7/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_7/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_7/moving_variance*
dtype0*
_output_shapes
: 
ś
6batch_normalization_7/moving_variance/Initializer/onesFillFbatch_normalization_7/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_7/moving_variance/Initializer/ones/Const*
_output_shapes	
:*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_7/moving_variance
Í
%batch_normalization_7/moving_variance
VariableV2*
_output_shapes	
:*
shared_name *8
_class.
,*loc:@batch_normalization_7/moving_variance*
	container *
shape:*
dtype0

,batch_normalization_7/moving_variance/AssignAssign%batch_normalization_7/moving_variance6batch_normalization_7/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
validate_shape(*
_output_shapes	
:
˝
*batch_normalization_7/moving_variance/readIdentity%batch_normalization_7/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
l
!batch_normalization_7/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_7/cond/switch_tIdentity#batch_normalization_7/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_7/cond/switch_fIdentity!batch_normalization_7/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_7/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_7/cond/ConstConst$^batch_normalization_7/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_7/cond/Const_1Const$^batch_normalization_7/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
´
)batch_normalization_7/cond/FusedBatchNormFusedBatchNorm2batch_normalization_7/cond/FusedBatchNorm/Switch:14batch_normalization_7/cond/FusedBatchNorm/Switch_1:14batch_normalization_7/cond/FusedBatchNorm/Switch_2:1 batch_normalization_7/cond/Const"batch_normalization_7/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training(
Ü
0batch_normalization_7/cond/FusedBatchNorm/SwitchSwitchConv2D_7"batch_normalization_7/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd*
T0*
_class
loc:@Conv2D_7
ß
2batch_normalization_7/cond/FusedBatchNorm/Switch_1Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_7/gamma*"
_output_shapes
::
Ý
2batch_normalization_7/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*-
_class#
!loc:@batch_normalization_7/beta*"
_output_shapes
::*
T0
Ü
+batch_normalization_7/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_7/cond/FusedBatchNorm_1/Switch4batch_normalization_7/cond/FusedBatchNorm_1/Switch_14batch_normalization_7/cond/FusedBatchNorm_1/Switch_24batch_normalization_7/cond/FusedBatchNorm_1/Switch_34batch_normalization_7/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( 
Ţ
2batch_normalization_7/cond/FusedBatchNorm_1/SwitchSwitchConv2D_7"batch_normalization_7/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd*
T0*
_class
loc:@Conv2D_7
á
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*"
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_7/gamma
ß
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*"
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_7/beta
í
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_7/moving_mean/read"batch_normalization_7/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*"
_output_shapes
::
ő
4batch_normalization_7/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_7/moving_variance/read"batch_normalization_7/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*"
_output_shapes
::
Ç
 batch_normalization_7/cond/MergeMerge+batch_normalization_7/cond/FusedBatchNorm_1)batch_normalization_7/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 
¸
"batch_normalization_7/cond/Merge_1Merge-batch_normalization_7/cond/FusedBatchNorm_1:1+batch_normalization_7/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes
	:: 
¸
"batch_normalization_7/cond/Merge_2Merge-batch_normalization_7/cond/FusedBatchNorm_1:2+batch_normalization_7/cond/FusedBatchNorm:2*
N*
_output_shapes
	:: *
T0
n
#batch_normalization_7/cond_1/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

{
%batch_normalization_7/cond_1/switch_tIdentity%batch_normalization_7/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_7/cond_1/switch_fIdentity#batch_normalization_7/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_7/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_7/cond_1/ConstConst&^batch_normalization_7/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=

$batch_normalization_7/cond_1/Const_1Const&^batch_normalization_7/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_7/cond_1/MergeMerge$batch_normalization_7/cond_1/Const_1"batch_normalization_7/cond_1/Const*
T0*
N*
_output_shapes
: : 
ľ
*batch_normalization_7/AssignMovingAvg/readIdentity!batch_normalization_7/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
Ü
)batch_normalization_7/AssignMovingAvg/SubSub*batch_normalization_7/AssignMovingAvg/read"batch_normalization_7/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
Ű
)batch_normalization_7/AssignMovingAvg/MulMul)batch_normalization_7/AssignMovingAvg/Sub"batch_normalization_7/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
ď
%batch_normalization_7/AssignMovingAvg	AssignSub!batch_normalization_7/moving_mean)batch_normalization_7/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:
ż
,batch_normalization_7/AssignMovingAvg_1/readIdentity%batch_normalization_7/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
ä
+batch_normalization_7/AssignMovingAvg_1/SubSub,batch_normalization_7/AssignMovingAvg_1/read"batch_normalization_7/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
ă
+batch_normalization_7/AssignMovingAvg_1/MulMul+batch_normalization_7/AssignMovingAvg_1/Sub"batch_normalization_7/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:
ű
'batch_normalization_7/AssignMovingAvg_1	AssignSub%batch_normalization_7/moving_variance+batch_normalization_7/AssignMovingAvg_1/Mul*
_output_shapes	
:*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance

	BiasAdd_7BiasAdd batch_normalization_7/cond/MergeVariable_15/read*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
m
ResizeNearestNeighbor_1/sizeConst*
valueB"Č   Č   *
dtype0*
_output_shapes
:
Ť
ResizeNearestNeighbor_1ResizeNearestNeighbor	BiasAdd_7ResizeNearestNeighbor_1/size*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
align_corners( 
q
truncated_normal_8/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
\
truncated_normal_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_8/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
§
"truncated_normal_8/TruncatedNormalTruncatedNormaltruncated_normal_8/shape*
dtype0*'
_output_shapes
:@*
seed2 *

seed *
T0

truncated_normal_8/mulMul"truncated_normal_8/TruncatedNormaltruncated_normal_8/stddev*'
_output_shapes
:@*
T0
|
truncated_normal_8Addtruncated_normal_8/multruncated_normal_8/mean*'
_output_shapes
:@*
T0

Variable_16
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@*
	container *
shape:@
¸
Variable_16/AssignAssignVariable_16truncated_normal_8*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*'
_output_shapes
:@
{
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16*'
_output_shapes
:@
T
Const_8Const*
valueB@*ÍĚĚ=*
dtype0*
_output_shapes
:@
w
Variable_17
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
 
Variable_17/AssignAssignVariable_17Const_8*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:@
n
Variable_17/readIdentityVariable_17*
_class
loc:@Variable_17*
_output_shapes
:@*
T0
ç
Conv2D_8Conv2DResizeNearestNeighbor_1Variable_16/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
	dilations
*
T0
ś
<batch_normalization_8/gamma/Initializer/ones/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_8/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?*.
_class$
" loc:@batch_normalization_8/gamma

,batch_normalization_8/gamma/Initializer/onesFill<batch_normalization_8/gamma/Initializer/ones/shape_as_tensor2batch_normalization_8/gamma/Initializer/ones/Const*
_output_shapes
:@*
T0*

index_type0*.
_class$
" loc:@batch_normalization_8/gamma
ˇ
batch_normalization_8/gamma
VariableV2*.
_class$
" loc:@batch_normalization_8/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
ő
"batch_normalization_8/gamma/AssignAssignbatch_normalization_8/gamma,batch_normalization_8/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
validate_shape(*
_output_shapes
:@

 batch_normalization_8/gamma/readIdentitybatch_normalization_8/gamma*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
ľ
<batch_normalization_8/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:@*-
_class#
!loc:@batch_normalization_8/beta*
dtype0
Ś
2batch_normalization_8/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_8/beta

,batch_normalization_8/beta/Initializer/zerosFill<batch_normalization_8/beta/Initializer/zeros/shape_as_tensor2batch_normalization_8/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
ľ
batch_normalization_8/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_8/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
ň
!batch_normalization_8/beta/AssignAssignbatch_normalization_8/beta,batch_normalization_8/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_8/beta

batch_normalization_8/beta/readIdentitybatch_normalization_8/beta*
T0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
Ă
Cbatch_normalization_8/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:@*4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_8/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0
¨
3batch_normalization_8/moving_mean/Initializer/zerosFillCbatch_normalization_8/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_8/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
Ă
!batch_normalization_8/moving_mean
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *4
_class*
(&loc:@batch_normalization_8/moving_mean

(batch_normalization_8/moving_mean/AssignAssign!batch_normalization_8/moving_mean3batch_normalization_8/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
validate_shape(*
_output_shapes
:@
°
&batch_normalization_8/moving_mean/readIdentity!batch_normalization_8/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
Ę
Fbatch_normalization_8/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:@*8
_class.
,*loc:@batch_normalization_8/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_8/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_8/moving_variance*
dtype0*
_output_shapes
: 
ľ
6batch_normalization_8/moving_variance/Initializer/onesFillFbatch_normalization_8/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_8/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@
Ë
%batch_normalization_8/moving_variance
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *8
_class.
,*loc:@batch_normalization_8/moving_variance

,batch_normalization_8/moving_variance/AssignAssign%batch_normalization_8/moving_variance6batch_normalization_8/moving_variance/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
validate_shape(*
_output_shapes
:@
ź
*batch_normalization_8/moving_variance/readIdentity%batch_normalization_8/moving_variance*
_output_shapes
:@*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance
l
!batch_normalization_8/cond/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
w
#batch_normalization_8/cond/switch_tIdentity#batch_normalization_8/cond/Switch:1*
_output_shapes
:*
T0

u
#batch_normalization_8/cond/switch_fIdentity!batch_normalization_8/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_8/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_8/cond/ConstConst$^batch_normalization_8/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_8/cond/Const_1Const$^batch_normalization_8/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
ą
)batch_normalization_8/cond/FusedBatchNormFusedBatchNorm2batch_normalization_8/cond/FusedBatchNorm/Switch:14batch_normalization_8/cond/FusedBatchNorm/Switch_1:14batch_normalization_8/cond/FusedBatchNorm/Switch_2:1 batch_normalization_8/cond/Const"batch_normalization_8/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training(
Ţ
0batch_normalization_8/cond/FusedBatchNorm/SwitchSwitchConv2D_8"batch_normalization_8/cond/pred_id*
T0*
_class
loc:@Conv2D_8*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
Ý
2batch_normalization_8/cond/FusedBatchNorm/Switch_1Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_8/gamma* 
_output_shapes
:@:@
Ű
2batch_normalization_8/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_8/beta* 
_output_shapes
:@:@
Ů
+batch_normalization_8/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_8/cond/FusedBatchNorm_1/Switch4batch_normalization_8/cond/FusedBatchNorm_1/Switch_14batch_normalization_8/cond/FusedBatchNorm_1/Switch_24batch_normalization_8/cond/FusedBatchNorm_1/Switch_34batch_normalization_8/cond/FusedBatchNorm_1/Switch_4*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( *
epsilon%o:*
T0*
data_formatNHWC
ŕ
2batch_normalization_8/cond/FusedBatchNorm_1/SwitchSwitchConv2D_8"batch_normalization_8/cond/pred_id*
_class
loc:@Conv2D_8*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0
ß
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_8/gamma* 
_output_shapes
:@:@
Ý
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_8/beta* 
_output_shapes
:@:@
ë
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_8/moving_mean/read"batch_normalization_8/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean* 
_output_shapes
:@:@
ó
4batch_normalization_8/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_8/moving_variance/read"batch_normalization_8/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance* 
_output_shapes
:@:@
Č
 batch_normalization_8/cond/MergeMerge+batch_normalization_8/cond/FusedBatchNorm_1)batch_normalization_8/cond/FusedBatchNorm*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 
ˇ
"batch_normalization_8/cond/Merge_1Merge-batch_normalization_8/cond/FusedBatchNorm_1:1+batch_normalization_8/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:@: 
ˇ
"batch_normalization_8/cond/Merge_2Merge-batch_normalization_8/cond/FusedBatchNorm_1:2+batch_normalization_8/cond/FusedBatchNorm:2*
N*
_output_shapes

:@: *
T0
n
#batch_normalization_8/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_8/cond_1/switch_tIdentity%batch_normalization_8/cond_1/Switch:1*
T0
*
_output_shapes
:
y
%batch_normalization_8/cond_1/switch_fIdentity#batch_normalization_8/cond_1/Switch*
T0
*
_output_shapes
:
`
$batch_normalization_8/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_8/cond_1/ConstConst&^batch_normalization_8/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ÍĚĚ=

$batch_normalization_8/cond_1/Const_1Const&^batch_normalization_8/cond_1/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ą
"batch_normalization_8/cond_1/MergeMerge$batch_normalization_8/cond_1/Const_1"batch_normalization_8/cond_1/Const*
N*
_output_shapes
: : *
T0
´
*batch_normalization_8/AssignMovingAvg/readIdentity!batch_normalization_8/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
Ű
)batch_normalization_8/AssignMovingAvg/SubSub*batch_normalization_8/AssignMovingAvg/read"batch_normalization_8/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
Ú
)batch_normalization_8/AssignMovingAvg/MulMul)batch_normalization_8/AssignMovingAvg/Sub"batch_normalization_8/cond_1/Merge*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean
î
%batch_normalization_8/AssignMovingAvg	AssignSub!batch_normalization_8/moving_mean)batch_normalization_8/AssignMovingAvg/Mul*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
:@
ž
,batch_normalization_8/AssignMovingAvg_1/readIdentity%batch_normalization_8/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@
ă
+batch_normalization_8/AssignMovingAvg_1/SubSub,batch_normalization_8/AssignMovingAvg_1/read"batch_normalization_8/cond/Merge_2*
_output_shapes
:@*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance
â
+batch_normalization_8/AssignMovingAvg_1/MulMul+batch_normalization_8/AssignMovingAvg_1/Sub"batch_normalization_8/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@
ú
'batch_normalization_8/AssignMovingAvg_1	AssignSub%batch_normalization_8/moving_variance+batch_normalization_8/AssignMovingAvg_1/Mul*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
:@

	BiasAdd_8BiasAdd batch_normalization_8/cond/MergeVariable_17/read*
T0*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
m
ResizeNearestNeighbor_2/sizeConst*
valueB"    *
dtype0*
_output_shapes
:
Ş
ResizeNearestNeighbor_2ResizeNearestNeighbor	BiasAdd_8ResizeNearestNeighbor_2/size*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
align_corners( *
T0
q
truncated_normal_9/shapeConst*%
valueB"      @       *
dtype0*
_output_shapes
:
\
truncated_normal_9/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_9/stddevConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
Ś
"truncated_normal_9/TruncatedNormalTruncatedNormaltruncated_normal_9/shape*
T0*
dtype0*&
_output_shapes
:@ *
seed2 *

seed 

truncated_normal_9/mulMul"truncated_normal_9/TruncatedNormaltruncated_normal_9/stddev*
T0*&
_output_shapes
:@ 
{
truncated_normal_9Addtruncated_normal_9/multruncated_normal_9/mean*&
_output_shapes
:@ *
T0

Variable_18
VariableV2*
dtype0*&
_output_shapes
:@ *
	container *
shape:@ *
shared_name 
ˇ
Variable_18/AssignAssignVariable_18truncated_normal_9*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:@ 
z
Variable_18/readIdentityVariable_18*&
_output_shapes
:@ *
T0*
_class
loc:@Variable_18
T
Const_9Const*
valueB *ÍĚĚ=*
dtype0*
_output_shapes
: 
w
Variable_19
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
 
Variable_19/AssignAssignVariable_19Const_9*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
: 
n
Variable_19/readIdentityVariable_19*
_output_shapes
: *
T0*
_class
loc:@Variable_19
ç
Conv2D_9Conv2DResizeNearestNeighbor_2Variable_18/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ś
<batch_normalization_9/gamma/Initializer/ones/shape_as_tensorConst*
valueB: *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
:
§
2batch_normalization_9/gamma/Initializer/ones/ConstConst*
valueB
 *  ?*.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
: 

,batch_normalization_9/gamma/Initializer/onesFill<batch_normalization_9/gamma/Initializer/ones/shape_as_tensor2batch_normalization_9/gamma/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
ˇ
batch_normalization_9/gamma
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_9/gamma*
	container *
shape: 
ő
"batch_normalization_9/gamma/AssignAssignbatch_normalization_9/gamma,batch_normalization_9/gamma/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
validate_shape(*
_output_shapes
: 

 batch_normalization_9/gamma/readIdentitybatch_normalization_9/gamma*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_9/gamma
ľ
<batch_normalization_9/beta/Initializer/zeros/shape_as_tensorConst*
valueB: *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
:
Ś
2batch_normalization_9/beta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_9/beta*
dtype0

,batch_normalization_9/beta/Initializer/zerosFill<batch_normalization_9/beta/Initializer/zeros/shape_as_tensor2batch_normalization_9/beta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
ľ
batch_normalization_9/beta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_9/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
ň
!batch_normalization_9/beta/AssignAssignbatch_normalization_9/beta,batch_normalization_9/beta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_9/beta*
validate_shape(*
_output_shapes
: 

batch_normalization_9/beta/readIdentitybatch_normalization_9/beta*
T0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
Ă
Cbatch_normalization_9/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB: *4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0*
_output_shapes
:
´
9batch_normalization_9/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0*
_output_shapes
: 
¨
3batch_normalization_9/moving_mean/Initializer/zerosFillCbatch_normalization_9/moving_mean/Initializer/zeros/shape_as_tensor9batch_normalization_9/moving_mean/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ă
!batch_normalization_9/moving_mean
VariableV2*4
_class*
(&loc:@batch_normalization_9/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 

(batch_normalization_9/moving_mean/AssignAssign!batch_normalization_9/moving_mean3batch_normalization_9/moving_mean/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
validate_shape(*
_output_shapes
: 
°
&batch_normalization_9/moving_mean/readIdentity!batch_normalization_9/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ę
Fbatch_normalization_9/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB: *8
_class.
,*loc:@batch_normalization_9/moving_variance*
dtype0*
_output_shapes
:
ť
<batch_normalization_9/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ?*8
_class.
,*loc:@batch_normalization_9/moving_variance*
dtype0*
_output_shapes
: 
ľ
6batch_normalization_9/moving_variance/Initializer/onesFillFbatch_normalization_9/moving_variance/Initializer/ones/shape_as_tensor<batch_normalization_9/moving_variance/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes
: 
Ë
%batch_normalization_9/moving_variance
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *8
_class.
,*loc:@batch_normalization_9/moving_variance*
	container 

,batch_normalization_9/moving_variance/AssignAssign%batch_normalization_9/moving_variance6batch_normalization_9/moving_variance/Initializer/ones*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
ź
*batch_normalization_9/moving_variance/readIdentity%batch_normalization_9/moving_variance*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes
: 
l
!batch_normalization_9/cond/SwitchSwitchPlaceholderPlaceholder*
_output_shapes

::*
T0

w
#batch_normalization_9/cond/switch_tIdentity#batch_normalization_9/cond/Switch:1*
T0
*
_output_shapes
:
u
#batch_normalization_9/cond/switch_fIdentity!batch_normalization_9/cond/Switch*
T0
*
_output_shapes
:
^
"batch_normalization_9/cond/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

 batch_normalization_9/cond/ConstConst$^batch_normalization_9/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 

"batch_normalization_9/cond/Const_1Const$^batch_normalization_9/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
ą
)batch_normalization_9/cond/FusedBatchNormFusedBatchNorm2batch_normalization_9/cond/FusedBatchNorm/Switch:14batch_normalization_9/cond/FusedBatchNorm/Switch_1:14batch_normalization_9/cond/FusedBatchNorm/Switch_2:1 batch_normalization_9/cond/Const"batch_normalization_9/cond/Const_1*
epsilon%o:*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
is_training(
Ţ
0batch_normalization_9/cond/FusedBatchNorm/SwitchSwitchConv2D_9"batch_normalization_9/cond/pred_id*
T0*
_class
loc:@Conv2D_9*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
Ý
2batch_normalization_9/cond/FusedBatchNorm/Switch_1Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id*.
_class$
" loc:@batch_normalization_9/gamma* 
_output_shapes
: : *
T0
Ű
2batch_normalization_9/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_9/beta* 
_output_shapes
: : 
Ů
+batch_normalization_9/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_9/cond/FusedBatchNorm_1/Switch4batch_normalization_9/cond/FusedBatchNorm_1/Switch_14batch_normalization_9/cond/FusedBatchNorm_1/Switch_24batch_normalization_9/cond/FusedBatchNorm_1/Switch_34batch_normalization_9/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
is_training( 
ŕ
2batch_normalization_9/cond/FusedBatchNorm_1/SwitchSwitchConv2D_9"batch_normalization_9/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
T0*
_class
loc:@Conv2D_9
ß
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_9/gamma* 
_output_shapes
: : 
Ý
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_9/beta* 
_output_shapes
: : 
ë
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_9/moving_mean/read"batch_normalization_9/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean* 
_output_shapes
: : 
ó
4batch_normalization_9/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_9/moving_variance/read"batch_normalization_9/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance* 
_output_shapes
: : 
Č
 batch_normalization_9/cond/MergeMerge+batch_normalization_9/cond/FusedBatchNorm_1)batch_normalization_9/cond/FusedBatchNorm*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 
ˇ
"batch_normalization_9/cond/Merge_1Merge-batch_normalization_9/cond/FusedBatchNorm_1:1+batch_normalization_9/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

: : 
ˇ
"batch_normalization_9/cond/Merge_2Merge-batch_normalization_9/cond/FusedBatchNorm_1:2+batch_normalization_9/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

: : 
n
#batch_normalization_9/cond_1/SwitchSwitchPlaceholderPlaceholder*
T0
*
_output_shapes

::
{
%batch_normalization_9/cond_1/switch_tIdentity%batch_normalization_9/cond_1/Switch:1*
_output_shapes
:*
T0

y
%batch_normalization_9/cond_1/switch_fIdentity#batch_normalization_9/cond_1/Switch*
_output_shapes
:*
T0

`
$batch_normalization_9/cond_1/pred_idIdentityPlaceholder*
T0
*
_output_shapes
:

"batch_normalization_9/cond_1/ConstConst&^batch_normalization_9/cond_1/switch_t*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 

$batch_normalization_9/cond_1/Const_1Const&^batch_normalization_9/cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ą
"batch_normalization_9/cond_1/MergeMerge$batch_normalization_9/cond_1/Const_1"batch_normalization_9/cond_1/Const*
T0*
N*
_output_shapes
: : 
´
*batch_normalization_9/AssignMovingAvg/readIdentity!batch_normalization_9/moving_mean*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ű
)batch_normalization_9/AssignMovingAvg/SubSub*batch_normalization_9/AssignMovingAvg/read"batch_normalization_9/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
Ú
)batch_normalization_9/AssignMovingAvg/MulMul)batch_normalization_9/AssignMovingAvg/Sub"batch_normalization_9/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
î
%batch_normalization_9/AssignMovingAvg	AssignSub!batch_normalization_9/moving_mean)batch_normalization_9/AssignMovingAvg/Mul*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: *
use_locking( *
T0
ž
,batch_normalization_9/AssignMovingAvg_1/readIdentity%batch_normalization_9/moving_variance*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance
ă
+batch_normalization_9/AssignMovingAvg_1/SubSub,batch_normalization_9/AssignMovingAvg_1/read"batch_normalization_9/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes
: 
â
+batch_normalization_9/AssignMovingAvg_1/MulMul+batch_normalization_9/AssignMovingAvg_1/Sub"batch_normalization_9/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance
ú
'batch_normalization_9/AssignMovingAvg_1	AssignSub%batch_normalization_9/moving_variance+batch_normalization_9/AssignMovingAvg_1/Mul*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes
: *
use_locking( *
T0

	BiasAdd_9BiasAdd batch_normalization_9/cond/MergeVariable_19/read*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
^
Variable_20/initial_valueConst*
_output_shapes
: *
valueB
 *  pB*
dtype0
o
Variable_20
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ž
Variable_20/AssignAssignVariable_20Variable_20/initial_value*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*
_output_shapes
: 
j
Variable_20/readIdentityVariable_20*
T0*
_class
loc:@Variable_20*
_output_shapes
: 
^
Variable_21/initial_valueConst*
valueB
 *   C*
dtype0*
_output_shapes
: 
o
Variable_21
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Ž
Variable_21/AssignAssignVariable_21Variable_21/initial_value*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
j
Variable_21/readIdentityVariable_21*
_output_shapes
: *
T0*
_class
loc:@Variable_21
r
truncated_normal_10/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
]
truncated_normal_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
truncated_normal_10/stddevConst*
_output_shapes
: *
valueB
 *ÍĚĚ=*
dtype0
¨
#truncated_normal_10/TruncatedNormalTruncatedNormaltruncated_normal_10/shape*
T0*
dtype0*&
_output_shapes
: *
seed2 *

seed 

truncated_normal_10/mulMul#truncated_normal_10/TruncatedNormaltruncated_normal_10/stddev*
T0*&
_output_shapes
: 
~
truncated_normal_10Addtruncated_normal_10/multruncated_normal_10/mean*&
_output_shapes
: *
T0

Variable_22
VariableV2*&
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
¸
Variable_22/AssignAssignVariable_22truncated_normal_10*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
: 
z
Variable_22/readIdentityVariable_22*
T0*
_class
loc:@Variable_22*&
_output_shapes
: 
U
Const_10Const*
valueB*ÍĚĚ=*
dtype0*
_output_shapes
:
w
Variable_23
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ą
Variable_23/AssignAssignVariable_23Const_10*
_class
loc:@Variable_23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
n
Variable_23/readIdentityVariable_23*
T0*
_class
loc:@Variable_23*
_output_shapes
:
Ú
	Conv2D_10Conv2D	BiasAdd_9Variable_22/read*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
c
mulMul	Conv2D_10Variable_20/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
addAddmulVariable_21/read*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
N
subSubyadd*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
norm/mulMulsubsub*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c

norm/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
s
norm/SumSumnorm/mul
norm/Const*&
_output_shapes
:*

Tidx0*
	keep_dims(*
T0
L
	norm/SqrtSqrtnorm/Sum*
T0*&
_output_shapes
:
W
norm/SqueezeSqueeze	norm/Sqrt*
squeeze_dims
 *
T0*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
O
lossScalarSummary	loss/tagsnorm/Squeeze*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
z
!gradients/norm/Squeeze_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
 
#gradients/norm/Squeeze_grad/ReshapeReshapegradients/Fill!gradients/norm/Squeeze_grad/Shape*&
_output_shapes
:*
T0*
Tshape0

!gradients/norm/Sqrt_grad/SqrtGradSqrtGrad	norm/Sqrt#gradients/norm/Squeeze_grad/Reshape*&
_output_shapes
:*
T0
~
%gradients/norm/Sum_grad/Reshape/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
ł
gradients/norm/Sum_grad/ReshapeReshape!gradients/norm/Sqrt_grad/SqrtGrad%gradients/norm/Sum_grad/Reshape/shape*&
_output_shapes
:*
T0*
Tshape0
e
gradients/norm/Sum_grad/ShapeShapenorm/mul*
T0*
out_type0*
_output_shapes
:
˛
gradients/norm/Sum_grad/TileTilegradients/norm/Sum_grad/Reshapegradients/norm/Sum_grad/Shape*

Tmultiples0*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
gradients/norm/mul_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
b
gradients/norm/mul_grad/Shape_1Shapesub*
T0*
out_type0*
_output_shapes
:
Ă
-gradients/norm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/norm/mul_grad/Shapegradients/norm/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/norm/mul_grad/mulMulgradients/norm/Sum_grad/Tilesub*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
gradients/norm/mul_grad/SumSumgradients/norm/mul_grad/mul-gradients/norm/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
°
gradients/norm/mul_grad/ReshapeReshapegradients/norm/mul_grad/Sumgradients/norm/mul_grad/Shape*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/norm/mul_grad/mul_1Mulsubgradients/norm/Sum_grad/Tile*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
gradients/norm/mul_grad/Sum_1Sumgradients/norm/mul_grad/mul_1/gradients/norm/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ś
!gradients/norm/mul_grad/Reshape_1Reshapegradients/norm/mul_grad/Sum_1gradients/norm/mul_grad/Shape_1*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
v
(gradients/norm/mul_grad/tuple/group_depsNoOp ^gradients/norm/mul_grad/Reshape"^gradients/norm/mul_grad/Reshape_1
ř
0gradients/norm/mul_grad/tuple/control_dependencyIdentitygradients/norm/mul_grad/Reshape)^gradients/norm/mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/norm/mul_grad/Reshape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ţ
2gradients/norm/mul_grad/tuple/control_dependency_1Identity!gradients/norm/mul_grad/Reshape_1)^gradients/norm/mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/norm/mul_grad/Reshape_1*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
gradients/AddNAddN0gradients/norm/mul_grad/tuple/control_dependency2gradients/norm/mul_grad/tuple/control_dependency_1*
T0*2
_class(
&$loc:@gradients/norm/mul_grad/Reshape*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
gradients/sub_grad/ShapeShapey*
T0*
out_type0*
_output_shapes
:
]
gradients/sub_grad/Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/sub_grad/SumSumgradients/AddN(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ą
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/sub_grad/Sum_1Sumgradients/AddN*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ľ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
ä
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
ę
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
[
gradients/add_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
]
gradients/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ś
gradients/add_grad/SumSum-gradients/sub_grad/tuple/control_dependency_1(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ą
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
gradients/add_grad/Sum_1Sum-gradients/sub_grad/tuple/control_dependency_1*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
ä
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
a
gradients/mul_grad/ShapeShape	Conv2D_10*
out_type0*
_output_shapes
:*
T0
]
gradients/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
´
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

gradients/mul_grad/mulMul+gradients/add_grad/tuple/control_dependencyVariable_20/read*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ą
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

gradients/mul_grad/mul_1Mul	Conv2D_10+gradients/add_grad/tuple/control_dependency*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
ä
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
_output_shapes
: 

gradients/Conv2D_10_grad/ShapeNShapeN	BiasAdd_9Variable_22/read*
T0*
out_type0*
N* 
_output_shapes
::
w
gradients/Conv2D_10_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
ć
,gradients/Conv2D_10_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_10_grad/ShapeNVariable_22/read+gradients/mul_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
ź
-gradients/Conv2D_10_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_9gradients/Conv2D_10_grad/Const+gradients/mul_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 

)gradients/Conv2D_10_grad/tuple/group_depsNoOp-^gradients/Conv2D_10_grad/Conv2DBackpropInput.^gradients/Conv2D_10_grad/Conv2DBackpropFilter

1gradients/Conv2D_10_grad/tuple/control_dependencyIdentity,gradients/Conv2D_10_grad/Conv2DBackpropInput*^gradients/Conv2D_10_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

3gradients/Conv2D_10_grad/tuple/control_dependency_1Identity-gradients/Conv2D_10_grad/Conv2DBackpropFilter*^gradients/Conv2D_10_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_10_grad/Conv2DBackpropFilter*&
_output_shapes
: 
˘
$gradients/BiasAdd_9_grad/BiasAddGradBiasAddGrad1gradients/Conv2D_10_grad/tuple/control_dependency*
_output_shapes
: *
T0*
data_formatNHWC

)gradients/BiasAdd_9_grad/tuple/group_depsNoOp2^gradients/Conv2D_10_grad/tuple/control_dependency%^gradients/BiasAdd_9_grad/BiasAddGrad

1gradients/BiasAdd_9_grad/tuple/control_dependencyIdentity1gradients/Conv2D_10_grad/tuple/control_dependency*^gradients/BiasAdd_9_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ď
3gradients/BiasAdd_9_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_9_grad/BiasAddGrad*^gradients/BiasAdd_9_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_9_grad/BiasAddGrad*
_output_shapes
: 
´
9gradients/batch_normalization_9/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_9_grad/tuple/control_dependency"batch_normalization_9/cond/pred_id*
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 

@gradients/batch_normalization_9/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_9/cond/Merge_grad/cond_grad
Ď
Hgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_9/cond/Merge_grad/cond_gradA^gradients/batch_normalization_9/cond/Merge_grad/tuple/group_deps*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput
Ó
Jgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_9/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_9/cond/Merge_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_10_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
u
gradients/zeros_like	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:1*
_output_shapes
: *
T0
w
gradients/zeros_like_1	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:2*
T0*
_output_shapes
: 
w
gradients/zeros_like_2	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:3*
_output_shapes
: *
T0
w
gradients/zeros_like_3	ZerosLike-batch_normalization_9/cond/FusedBatchNorm_1:4*
T0*
_output_shapes
: 

Mgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependency2batch_normalization_9/cond/FusedBatchNorm_1/Switch4batch_normalization_9/cond/FusedBatchNorm_1/Switch_14batch_normalization_9/cond/FusedBatchNorm_1/Switch_34batch_normalization_9/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ : : : : *
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Ugradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 

Ugradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 
u
gradients/zeros_like_4	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:1*
T0*
_output_shapes
: 
u
gradients/zeros_like_5	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:2*
T0*
_output_shapes
: 
u
gradients/zeros_like_6	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:3*
_output_shapes
: *
T0
u
gradients/zeros_like_7	ZerosLike+batch_normalization_9/cond/FusedBatchNorm:4*
_output_shapes
: *
T0
ý
Kgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_9/cond/Merge_grad/tuple/control_dependency_12batch_normalization_9/cond/FusedBatchNorm/Switch:14batch_normalization_9/cond/FusedBatchNorm/Switch_1:1+batch_normalization_9/cond/FusedBatchNorm:3+batch_normalization_9/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ : : : : *
is_training(

Igradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
˙
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad
˙
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ý
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ą
gradients/SwitchSwitchConv2D_9"batch_normalization_9/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ *
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

Kgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 

gradients/Switch_1Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id*
T0* 
_output_shapes
: : 
e
gradients/Shape_2Shapegradients/Switch_1:1*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
|
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*

index_type0*
_output_shapes
: *
T0
đ
Mgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_1*
T0*
N*
_output_shapes

: : 

gradients/Switch_2Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id*
T0* 
_output_shapes
: : 
e
gradients/Shape_3Shapegradients/Switch_2:1*
out_type0*
_output_shapes
:*
T0
\
gradients/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zeros_2Fillgradients/Shape_3gradients/zeros_2/Const*
T0*

index_type0*
_output_shapes
: 
đ
Mgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_9/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_2*
_output_shapes

: : *
T0*
N
Ł
gradients/Switch_3SwitchConv2D_9"batch_normalization_9/cond/pred_id*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ :˙˙˙˙˙˙˙˙˙ 
c
gradients/Shape_4Shapegradients/Switch_3*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_3Fillgradients/Shape_4gradients/zeros_3/Const*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*

index_type0
˙
Igradients/batch_normalization_9/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_3Qgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ : 

gradients/Switch_4Switch batch_normalization_9/gamma/read"batch_normalization_9/cond/pred_id* 
_output_shapes
: : *
T0
c
gradients/Shape_5Shapegradients/Switch_4*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zeros_4Fillgradients/Shape_5gradients/zeros_4/Const*
T0*

index_type0*
_output_shapes
: 
ě
Kgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_4Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes

: : 

gradients/Switch_5Switchbatch_normalization_9/beta/read"batch_normalization_9/cond/pred_id*
T0* 
_output_shapes
: : 
c
gradients/Shape_6Shapegradients/Switch_5*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_5/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
|
gradients/zeros_5Fillgradients/Shape_6gradients/zeros_5/Const*
_output_shapes
: *
T0*

index_type0
ě
Kgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_5Sgradients/batch_normalization_9/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes

: : 
Ő
gradients/AddN_1AddNKgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

gradients/Conv2D_9_grad/ShapeNShapeNResizeNearestNeighbor_2Variable_18/read*
out_type0*
N* 
_output_shapes
::*
T0
v
gradients/Conv2D_9_grad/ConstConst*%
valueB"      @       *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_9_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_9_grad/ShapeNVariable_18/readgradients/AddN_1*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
­
,gradients/Conv2D_9_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighbor_2gradients/Conv2D_9_grad/Constgradients/AddN_1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@ *
	dilations
*
T0

(gradients/Conv2D_9_grad/tuple/group_depsNoOp,^gradients/Conv2D_9_grad/Conv2DBackpropInput-^gradients/Conv2D_9_grad/Conv2DBackpropFilter

0gradients/Conv2D_9_grad/tuple/control_dependencyIdentity+gradients/Conv2D_9_grad/Conv2DBackpropInput)^gradients/Conv2D_9_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_9_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙@

2gradients/Conv2D_9_grad/tuple/control_dependency_1Identity,gradients/Conv2D_9_grad/Conv2DBackpropFilter)^gradients/Conv2D_9_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_9_grad/Conv2DBackpropFilter*&
_output_shapes
:@ *
T0
Ä
gradients/AddN_2AddNMgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
: 
Ä
gradients/AddN_3AddNMgradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_9/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_9/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
: 

Egradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"Č   Č   *
dtype0*
_output_shapes
:
§
@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad0gradients/Conv2D_9_grad/tuple/control_dependencyEgradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad/size*
align_corners( *
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ą
$gradients/BiasAdd_8_grad/BiasAddGradBiasAddGrad@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*
T0*
data_formatNHWC*
_output_shapes
:@

)gradients/BiasAdd_8_grad/tuple/group_depsNoOpA^gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad%^gradients/BiasAdd_8_grad/BiasAddGrad
ź
1gradients/BiasAdd_8_grad/tuple/control_dependencyIdentity@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*^gradients/BiasAdd_8_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ď
3gradients/BiasAdd_8_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_8_grad/BiasAddGrad*^gradients/BiasAdd_8_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_8_grad/BiasAddGrad*
_output_shapes
:@
Č
9gradients/batch_normalization_8/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_8_grad/tuple/control_dependency"batch_normalization_8/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad

@gradients/batch_normalization_8/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_8/cond/Merge_grad/cond_grad
ă
Hgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_8/cond/Merge_grad/cond_gradA^gradients/batch_normalization_8/cond/Merge_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ç
Jgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_8/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_8/cond/Merge_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_2_grad/ResizeNearestNeighborGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
w
gradients/zeros_like_8	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:1*
T0*
_output_shapes
:@
w
gradients/zeros_like_9	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:2*
T0*
_output_shapes
:@
x
gradients/zeros_like_10	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:3*
T0*
_output_shapes
:@
x
gradients/zeros_like_11	ZerosLike-batch_normalization_8/cond/FusedBatchNorm_1:4*
T0*
_output_shapes
:@

Mgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependency2batch_normalization_8/cond/FusedBatchNorm_1/Switch4batch_normalization_8/cond/FusedBatchNorm_1/Switch_14batch_normalization_8/cond/FusedBatchNorm_1/Switch_34batch_normalization_8/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( 
Ł
Kgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

Ugradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@

Ugradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@
v
gradients/zeros_like_12	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:1*
T0*
_output_shapes
:@
v
gradients/zeros_like_13	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:2*
_output_shapes
:@*
T0
v
gradients/zeros_like_14	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:3*
T0*
_output_shapes
:@
v
gradients/zeros_like_15	ZerosLike+batch_normalization_8/cond/FusedBatchNorm:4*
T0*
_output_shapes
:@
ý
Kgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_8/cond/Merge_grad/tuple/control_dependency_12batch_normalization_8/cond/FusedBatchNorm/Switch:14batch_normalization_8/cond/FusedBatchNorm/Switch_1:1+batch_normalization_8/cond/FusedBatchNorm:3+batch_normalization_8/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ČČ@:@:@: : *
is_training(*
epsilon%o:

Igradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
˙
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
˙
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
ý
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ł
gradients/Switch_6SwitchConv2D_8"batch_normalization_8/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0
e
gradients/Shape_7Shapegradients/Switch_6:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_6Fillgradients/Shape_7gradients/zeros_6/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

Kgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_6*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: *
T0

gradients/Switch_7Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_8Shapegradients/Switch_7:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_7/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zeros_7Fillgradients/Shape_8gradients/zeros_7/Const*
T0*

index_type0*
_output_shapes
:@
đ
Mgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_7*
T0*
N*
_output_shapes

:@: 

gradients/Switch_8Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id* 
_output_shapes
:@:@*
T0
e
gradients/Shape_9Shapegradients/Switch_8:1*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zeros_8Fillgradients/Shape_9gradients/zeros_8/Const*
T0*

index_type0*
_output_shapes
:@
đ
Mgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_8/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_8*
T0*
N*
_output_shapes

:@: 
Ł
gradients/Switch_9SwitchConv2D_8"batch_normalization_8/cond/pred_id*
T0*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@
d
gradients/Shape_10Shapegradients/Switch_9*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_9/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_9Fillgradients/Shape_10gradients/zeros_9/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
˙
Igradients/batch_normalization_8/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_9Qgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: *
T0*
N

gradients/Switch_10Switch batch_normalization_8/gamma/read"batch_normalization_8/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_11Shapegradients/Switch_10*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_10Fillgradients/Shape_11gradients/zeros_10/Const*
_output_shapes
:@*
T0*

index_type0
í
Kgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_10Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes

:@: 

gradients/Switch_11Switchbatch_normalization_8/beta/read"batch_normalization_8/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_12Shapegradients/Switch_11*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_11/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_11Fillgradients/Shape_12gradients/zeros_11/Const*
T0*

index_type0*
_output_shapes
:@
í
Kgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_11Sgradients/batch_normalization_8/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes

:@: 
Ő
gradients/AddN_4AddNKgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
T0*^
_classT
RPloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_grad/cond_grad

gradients/Conv2D_8_grad/ShapeNShapeNResizeNearestNeighbor_1Variable_16/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_8_grad/ConstConst*
_output_shapes
:*%
valueB"         @   *
dtype0
É
+gradients/Conv2D_8_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_8_grad/ShapeNVariable_16/readgradients/AddN_4*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ž
,gradients/Conv2D_8_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighbor_1gradients/Conv2D_8_grad/Constgradients/AddN_4*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_8_grad/tuple/group_depsNoOp,^gradients/Conv2D_8_grad/Conv2DBackpropInput-^gradients/Conv2D_8_grad/Conv2DBackpropFilter

0gradients/Conv2D_8_grad/tuple/control_dependencyIdentity+gradients/Conv2D_8_grad/Conv2DBackpropInput)^gradients/Conv2D_8_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_8_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

2gradients/Conv2D_8_grad/tuple/control_dependency_1Identity,gradients/Conv2D_8_grad/Conv2DBackpropFilter)^gradients/Conv2D_8_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_8_grad/Conv2DBackpropFilter*'
_output_shapes
:@
Ä
gradients/AddN_5AddNMgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_1_grad/cond_grad*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:@*
T0
Ä
gradients/AddN_6AddNMgradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_8/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*
_output_shapes
:@*
T0*`
_classV
TRloc:@gradients/batch_normalization_8/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad

Egradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad/sizeConst*
_output_shapes
:*
valueB"d   d   *
dtype0
Ś
@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad0gradients/Conv2D_8_grad/tuple/control_dependencyEgradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad/size*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
align_corners( *
T0
˛
$gradients/BiasAdd_7_grad/BiasAddGradBiasAddGrad@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_7_grad/tuple/group_depsNoOpA^gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad%^gradients/BiasAdd_7_grad/BiasAddGrad
ť
1gradients/BiasAdd_7_grad/tuple/control_dependencyIdentity@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*^gradients/BiasAdd_7_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
đ
3gradients/BiasAdd_7_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_7_grad/BiasAddGrad*^gradients/BiasAdd_7_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_7_grad/BiasAddGrad*
_output_shapes	
:*
T0
Ć
9gradients/batch_normalization_7/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_7_grad/tuple/control_dependency"batch_normalization_7/cond/pred_id*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd

@gradients/batch_normalization_7/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_7/cond/Merge_grad/cond_grad
â
Hgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_7/cond/Merge_grad/cond_gradA^gradients/batch_normalization_7/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad
ć
Jgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_7/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_7/cond/Merge_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ResizeNearestNeighbor_1_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
y
gradients/zeros_like_16	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:1*
_output_shapes	
:*
T0
y
gradients/zeros_like_17	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_18	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_19	ZerosLike-batch_normalization_7/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependency2batch_normalization_7/cond/FusedBatchNorm_1/Switch4batch_normalization_7/cond/FusedBatchNorm_1/Switch_14batch_normalization_7/cond/FusedBatchNorm_1/Switch_34batch_normalization_7/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( 
Ł
Kgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Ugradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_20	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_21	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:2*
_output_shapes	
:*
T0
w
gradients/zeros_like_22	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_23	ZerosLike+batch_normalization_7/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_7/cond/Merge_grad/tuple/control_dependency_12batch_normalization_7/cond/FusedBatchNorm/Switch:14batch_normalization_7/cond/FusedBatchNorm/Switch_1:1+batch_normalization_7/cond/FusedBatchNorm:3+batch_normalization_7/cond/FusedBatchNorm:4*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙dd::: : *
is_training(*
epsilon%o:*
T0*
data_formatNHWC

Igradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:*
T0

Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ý
Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_12SwitchConv2D_7"batch_normalization_7/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
g
gradients/Shape_13Shapegradients/Switch_12:1*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_12Fillgradients/Shape_13gradients/zeros_12/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

Kgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_12*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 

gradients/Switch_13Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_14Shapegradients/Switch_13:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_13/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_13Fillgradients/Shape_14gradients/zeros_13/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_13*
T0*
N*
_output_shapes
	:: 

gradients/Switch_14Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_15Shapegradients/Switch_14:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_14Fillgradients/Shape_15gradients/zeros_14/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_7/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_14*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_15SwitchConv2D_7"batch_normalization_7/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
e
gradients/Shape_16Shapegradients/Switch_15*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_15/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_15Fillgradients/Shape_16gradients/zeros_15/Const*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0*

index_type0
˙
Igradients/batch_normalization_7/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_15Qgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 

gradients/Switch_16Switch batch_normalization_7/gamma/read"batch_normalization_7/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_17Shapegradients/Switch_16*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_16Fillgradients/Shape_17gradients/zeros_16/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_16Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
N*
_output_shapes
	:: *
T0

gradients/Switch_17Switchbatch_normalization_7/beta/read"batch_normalization_7/cond/pred_id*"
_output_shapes
::*
T0
e
gradients/Shape_18Shapegradients/Switch_17*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_17/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_17Fillgradients/Shape_18gradients/zeros_17/Const*
_output_shapes	
:*
T0*

index_type0
î
Kgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_17Sgradients/batch_normalization_7/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ô
gradients/AddN_7AddNKgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

gradients/Conv2D_7_grad/ShapeNShapeNResizeNearestNeighborVariable_14/read*
N* 
_output_shapes
::*
T0*
out_type0
v
gradients/Conv2D_7_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_7_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_7_grad/ShapeNVariable_14/readgradients/AddN_7*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
­
,gradients/Conv2D_7_grad/Conv2DBackpropFilterConv2DBackpropFilterResizeNearestNeighborgradients/Conv2D_7_grad/Constgradients/AddN_7*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/Conv2D_7_grad/tuple/group_depsNoOp,^gradients/Conv2D_7_grad/Conv2DBackpropInput-^gradients/Conv2D_7_grad/Conv2DBackpropFilter

0gradients/Conv2D_7_grad/tuple/control_dependencyIdentity+gradients/Conv2D_7_grad/Conv2DBackpropInput)^gradients/Conv2D_7_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_7_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

2gradients/Conv2D_7_grad/tuple/control_dependency_1Identity,gradients/Conv2D_7_grad/Conv2DBackpropFilter)^gradients/Conv2D_7_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_7_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ĺ
gradients/AddN_8AddNMgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N
Ĺ
gradients/AddN_9AddNMgradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_7/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_7/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N

Cgradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"2   2   *
dtype0*
_output_shapes
:
˘
>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad0gradients/Conv2D_7_grad/tuple/control_dependencyCgradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
align_corners( *
T0
°
$gradients/BiasAdd_6_grad/BiasAddGradBiasAddGrad>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_6_grad/tuple/group_depsNoOp?^gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad%^gradients/BiasAdd_6_grad/BiasAddGrad
ˇ
1gradients/BiasAdd_6_grad/tuple/control_dependencyIdentity>gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*^gradients/BiasAdd_6_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
đ
3gradients/BiasAdd_6_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_6_grad/BiasAddGrad*^gradients/BiasAdd_6_grad/tuple/group_deps*
_output_shapes	
:*
T0*7
_class-
+)loc:@gradients/BiasAdd_6_grad/BiasAddGrad
Ä
9gradients/batch_normalization_6/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_6_grad/tuple/control_dependency"batch_normalization_6/cond/pred_id*
T0*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22

@gradients/batch_normalization_6/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_6/cond/Merge_grad/cond_grad
ŕ
Hgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_6/cond/Merge_grad/cond_gradA^gradients/batch_normalization_6/cond/Merge_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0
ä
Jgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_6/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_6/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*Q
_classG
ECloc:@gradients/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad
y
gradients/zeros_like_24	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:1*
T0*
_output_shapes	
:
y
gradients/zeros_like_25	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_26	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_27	ZerosLike-batch_normalization_6/cond/FusedBatchNorm_1:4*
T0*
_output_shapes	
:

Mgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependency2batch_normalization_6/cond/FusedBatchNorm_1/Switch4batch_normalization_6/cond/FusedBatchNorm_1/Switch_14batch_normalization_6/cond/FusedBatchNorm_1/Switch_34batch_normalization_6/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( 
Ł
Kgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Ugradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_28	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_29	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_30	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:3*
_output_shapes	
:*
T0
w
gradients/zeros_like_31	ZerosLike+batch_normalization_6/cond/FusedBatchNorm:4*
_output_shapes	
:*
T0
ţ
Kgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_6/cond/Merge_grad/tuple/control_dependency_12batch_normalization_6/cond/FusedBatchNorm/Switch:14batch_normalization_6/cond/FusedBatchNorm/Switch_1:1+batch_normalization_6/cond/FusedBatchNorm:3+batch_normalization_6/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(*
epsilon%o:

Igradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ý
Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_18SwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
g
gradients/Shape_19Shapegradients/Switch_18:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_18Fillgradients/Shape_19gradients/zeros_18/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Kgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_18*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: *
T0*
N

gradients/Switch_19Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_20Shapegradients/Switch_19:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_19Fillgradients/Shape_20gradients/zeros_19/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_19*
T0*
N*
_output_shapes
	:: 

gradients/Switch_20Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_21Shapegradients/Switch_20:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_20/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_20Fillgradients/Shape_21gradients/zeros_20/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_6/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_20*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_21SwitchConv2D_6"batch_normalization_6/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
e
gradients/Shape_22Shapegradients/Switch_21*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_21/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_21Fillgradients/Shape_22gradients/zeros_21/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_6/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_21Qgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_22Switch batch_normalization_6/gamma/read"batch_normalization_6/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_23Shapegradients/Switch_22*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_22Fillgradients/Shape_23gradients/zeros_22/Const*
_output_shapes	
:*
T0*

index_type0
î
Kgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_22Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
_output_shapes
	:: *
T0*
N

gradients/Switch_23Switchbatch_normalization_6/beta/read"batch_normalization_6/cond/pred_id*"
_output_shapes
::*
T0
e
gradients/Shape_24Shapegradients/Switch_23*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_23/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_23Fillgradients/Shape_24gradients/zeros_23/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_23Sgradients/batch_normalization_6/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
_output_shapes
	:: *
T0*
N
Ő
gradients/AddN_10AddNKgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_grad/cond_grad*^
_classT
RPloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0

gradients/Conv2D_6_grad/ShapeNShapeN	BiasAdd_5Variable_12/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_6_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"            
Ę
+gradients/Conv2D_6_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_6_grad/ShapeNVariable_12/readgradients/AddN_10*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0
˘
,gradients/Conv2D_6_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_5gradients/Conv2D_6_grad/Constgradients/AddN_10*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_6_grad/tuple/group_depsNoOp,^gradients/Conv2D_6_grad/Conv2DBackpropInput-^gradients/Conv2D_6_grad/Conv2DBackpropFilter

0gradients/Conv2D_6_grad/tuple/control_dependencyIdentity+gradients/Conv2D_6_grad/Conv2DBackpropInput)^gradients/Conv2D_6_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

2gradients/Conv2D_6_grad/tuple/control_dependency_1Identity,gradients/Conv2D_6_grad/Conv2DBackpropFilter)^gradients/Conv2D_6_grad/tuple/group_deps*(
_output_shapes
:*
T0*?
_class5
31loc:@gradients/Conv2D_6_grad/Conv2DBackpropFilter
Ć
gradients/AddN_11AddNMgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
N*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad
Ć
gradients/AddN_12AddNMgradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_6/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_6/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_5_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_6_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_5_grad/tuple/group_depsNoOp1^gradients/Conv2D_6_grad/tuple/control_dependency%^gradients/BiasAdd_5_grad/BiasAddGrad

1gradients/BiasAdd_5_grad/tuple/control_dependencyIdentity0gradients/Conv2D_6_grad/tuple/control_dependency*^gradients/BiasAdd_5_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0
đ
3gradients/BiasAdd_5_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_5_grad/BiasAddGrad*^gradients/BiasAdd_5_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_5_grad/BiasAddGrad*
_output_shapes	
:
ą
9gradients/batch_normalization_5/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_5_grad/tuple/control_dependency"batch_normalization_5/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22

@gradients/batch_normalization_5/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_5/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_5/cond/Merge_grad/cond_gradA^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput
Ń
Jgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_5/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_5/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_6_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
y
gradients/zeros_like_32	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:1*
T0*
_output_shapes	
:
y
gradients/zeros_like_33	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_34	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_35	ZerosLike-batch_normalization_5/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency2batch_normalization_5/cond/FusedBatchNorm_1/Switch4batch_normalization_5/cond/FusedBatchNorm_1/Switch_14batch_normalization_5/cond/FusedBatchNorm_1/Switch_34batch_normalization_5/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/group_deps*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:*
T0
w
gradients/zeros_like_36	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:1*
_output_shapes	
:*
T0
w
gradients/zeros_like_37	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_38	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_39	ZerosLike+batch_normalization_5/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_5/cond/Merge_grad/tuple/control_dependency_12batch_normalization_5/cond/FusedBatchNorm/Switch:14batch_normalization_5/cond/FusedBatchNorm/Switch_1:1+batch_normalization_5/cond/FusedBatchNorm:3+batch_normalization_5/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(

Igradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
ý
Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_24SwitchConv2D_5"batch_normalization_5/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0
g
gradients/Shape_25Shapegradients/Switch_24:1*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_24/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_24Fillgradients/Shape_25gradients/zeros_24/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Kgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_24*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_25Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_26Shapegradients/Switch_25:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_25/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_25Fillgradients/Shape_26gradients/zeros_25/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_25*
N*
_output_shapes
	:: *
T0

gradients/Switch_26Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_27Shapegradients/Switch_26:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_26/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_26Fillgradients/Shape_27gradients/zeros_26/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_5/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_26*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_27SwitchConv2D_5"batch_normalization_5/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
e
gradients/Shape_28Shapegradients/Switch_27*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_27/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_27Fillgradients/Shape_28gradients/zeros_27/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_27Qgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_28Switch batch_normalization_5/gamma/read"batch_normalization_5/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_29Shapegradients/Switch_28*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_28Fillgradients/Shape_29gradients/zeros_28/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_28Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_29Switchbatch_normalization_5/beta/read"batch_normalization_5/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_30Shapegradients/Switch_29*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_29/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_29Fillgradients/Shape_30gradients/zeros_29/Const*

index_type0*
_output_shapes	
:*
T0
î
Kgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_29Sgradients/batch_normalization_5/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ő
gradients/AddN_13AddNKgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

gradients/Conv2D_5_grad/ShapeNShapeN	BiasAdd_4Variable_10/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_5_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
Ę
+gradients/Conv2D_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_5_grad/ShapeNVariable_10/readgradients/AddN_13*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0
˘
,gradients/Conv2D_5_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_4gradients/Conv2D_5_grad/Constgradients/AddN_13*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_5_grad/tuple/group_depsNoOp,^gradients/Conv2D_5_grad/Conv2DBackpropInput-^gradients/Conv2D_5_grad/Conv2DBackpropFilter

0gradients/Conv2D_5_grad/tuple/control_dependencyIdentity+gradients/Conv2D_5_grad/Conv2DBackpropInput)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

2gradients/Conv2D_5_grad/tuple/control_dependency_1Identity,gradients/Conv2D_5_grad/Conv2DBackpropFilter)^gradients/Conv2D_5_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_5_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ć
gradients/AddN_14AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_15AddNMgradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_5/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_5/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_5_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_4_grad/tuple/group_depsNoOp1^gradients/Conv2D_5_grad/tuple/control_dependency%^gradients/BiasAdd_4_grad/BiasAddGrad

1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentity0gradients/Conv2D_5_grad/tuple/control_dependency*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
đ
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes	
:
ą
9gradients/batch_normalization_4/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_4_grad/tuple/control_dependency"batch_normalization_4/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput

@gradients/batch_normalization_4/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_4/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_4/cond/Merge_grad/cond_gradA^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
Ń
Jgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_4/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_4/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_5_grad/Conv2DBackpropInput
y
gradients/zeros_like_40	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:1*
T0*
_output_shapes	
:
y
gradients/zeros_like_41	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:2*
_output_shapes	
:*
T0
y
gradients/zeros_like_42	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_43	ZerosLike-batch_normalization_4/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency2batch_normalization_4/cond/FusedBatchNorm_1/Switch4batch_normalization_4/cond/FusedBatchNorm_1/Switch_14batch_normalization_4/cond/FusedBatchNorm_1/Switch_34batch_normalization_4/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_44	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_45	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_46	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_47	ZerosLike+batch_normalization_4/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_4/cond/Merge_grad/tuple/control_dependency_12batch_normalization_4/cond/FusedBatchNorm/Switch:14batch_normalization_4/cond/FusedBatchNorm/Switch_1:1+batch_normalization_4/cond/FusedBatchNorm:3+batch_normalization_4/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(*
epsilon%o:

Igradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:

Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_grad/FusedBatchNormGrad
˘
gradients/Switch_30SwitchConv2D_4"batch_normalization_4/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
g
gradients/Shape_31Shapegradients/Switch_30:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_30Fillgradients/Shape_31gradients/zeros_30/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Kgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_30*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_31Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_32Shapegradients/Switch_31:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_31/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_31Fillgradients/Shape_32gradients/zeros_31/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_31*
T0*
N*
_output_shapes
	:: 

gradients/Switch_32Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*"
_output_shapes
::*
T0
g
gradients/Shape_33Shapegradients/Switch_32:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_32Fillgradients/Shape_33gradients/zeros_32/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_4/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_32*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_33SwitchConv2D_4"batch_normalization_4/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
e
gradients/Shape_34Shapegradients/Switch_33*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_33/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_33Fillgradients/Shape_34gradients/zeros_33/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_33Qgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: *
T0

gradients/Switch_34Switch batch_normalization_4/gamma/read"batch_normalization_4/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_35Shapegradients/Switch_34*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_34Fillgradients/Shape_35gradients/zeros_34/Const*

index_type0*
_output_shapes	
:*
T0
î
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_34Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_35Switchbatch_normalization_4/beta/read"batch_normalization_4/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_36Shapegradients/Switch_35*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_35/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_35Fillgradients/Shape_36gradients/zeros_35/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_35Sgradients/batch_normalization_4/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ő
gradients/AddN_16AddNKgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_grad/cond_grad*^
_classT
RPloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0

gradients/Conv2D_4_grad/ShapeNShapeN	BiasAdd_3Variable_8/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_4_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"            
É
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeNVariable_8/readgradients/AddN_16*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
˘
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_3gradients/Conv2D_4_grad/Constgradients/AddN_16*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:

(gradients/Conv2D_4_grad/tuple/group_depsNoOp,^gradients/Conv2D_4_grad/Conv2DBackpropInput-^gradients/Conv2D_4_grad/Conv2DBackpropFilter

0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput

2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*(
_output_shapes
:*
T0*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter
Ć
gradients/AddN_17AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_18AddNMgradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_4/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_4/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_4_grad/tuple/control_dependency*
_output_shapes	
:*
T0*
data_formatNHWC

)gradients/BiasAdd_3_grad/tuple/group_depsNoOp1^gradients/Conv2D_4_grad/tuple/control_dependency%^gradients/BiasAdd_3_grad/BiasAddGrad

1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentity0gradients/Conv2D_4_grad/tuple/control_dependency*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
đ
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad*
_output_shapes	
:
ą
9gradients/batch_normalization_3/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_3_grad/tuple/control_dependency"batch_normalization_3/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22

@gradients/batch_normalization_3/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_3/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_3/cond/Merge_grad/cond_gradA^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput
Ń
Jgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_3/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_3/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
y
gradients/zeros_like_48	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:1*
_output_shapes	
:*
T0
y
gradients/zeros_like_49	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_50	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_51	ZerosLike-batch_normalization_3/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency2batch_normalization_3/cond/FusedBatchNorm_1/Switch4batch_normalization_3/cond/FusedBatchNorm_1/Switch_14batch_normalization_3/cond/FusedBatchNorm_1/Switch_34batch_normalization_3/cond/FusedBatchNorm_1/Switch_4*
epsilon%o:*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22::::*
is_training( 
Ł
Kgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_52	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_53	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:2*
_output_shapes	
:*
T0
w
gradients/zeros_like_54	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_55	ZerosLike+batch_normalization_3/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_3/cond/Merge_grad/tuple/control_dependency_12batch_normalization_3/cond/FusedBatchNorm/Switch:14batch_normalization_3/cond/FusedBatchNorm/Switch_1:1+batch_normalization_3/cond/FusedBatchNorm:3+batch_normalization_3/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙22::: : *
is_training(*
epsilon%o:

Igradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:

Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_36SwitchConv2D_3"batch_normalization_3/cond/pred_id*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22*
T0
g
gradients/Shape_37Shapegradients/Switch_36:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_36Fillgradients/Shape_37gradients/zeros_36/Const*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22*
T0*

index_type0

Kgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_36*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_37Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_38Shapegradients/Switch_37:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_37/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_37Fillgradients/Shape_38gradients/zeros_37/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_37*
T0*
N*
_output_shapes
	:: 

gradients/Switch_38Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_39Shapegradients/Switch_38:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_38/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_38Fillgradients/Shape_39gradients/zeros_38/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_3/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_38*
T0*
N*
_output_shapes
	:: 
˘
gradients/Switch_39SwitchConv2D_3"batch_normalization_3/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙22:˙˙˙˙˙˙˙˙˙22
e
gradients/Shape_40Shapegradients/Switch_39*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_39/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_39Fillgradients/Shape_40gradients/zeros_39/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22
˙
Igradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_39Qgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙22: 

gradients/Switch_40Switch batch_normalization_3/gamma/read"batch_normalization_3/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_41Shapegradients/Switch_40*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_40/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_40Fillgradients/Shape_41gradients/zeros_40/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_40Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
N*
_output_shapes
	:: *
T0

gradients/Switch_41Switchbatch_normalization_3/beta/read"batch_normalization_3/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_42Shapegradients/Switch_41*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_41/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_41Fillgradients/Shape_42gradients/zeros_41/Const*

index_type0*
_output_shapes	
:*
T0
î
Kgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_41Sgradients/batch_normalization_3/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
Ő
gradients/AddN_19AddNKgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙22

gradients/Conv2D_3_grad/ShapeNShapeN	BiasAdd_2Variable_6/read*
out_type0*
N* 
_output_shapes
::*
T0
v
gradients/Conv2D_3_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeNVariable_6/readgradients/AddN_19*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
˘
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_2gradients/Conv2D_3_grad/Constgradients/AddN_19*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/Conv2D_3_grad/tuple/group_depsNoOp,^gradients/Conv2D_3_grad/Conv2DBackpropInput-^gradients/Conv2D_3_grad/Conv2DBackpropFilter

0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ć
gradients/AddN_20AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_21AddNMgradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_3/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
N*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/batch_normalization_3/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad
˘
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_3_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

)gradients/BiasAdd_2_grad/tuple/group_depsNoOp1^gradients/Conv2D_3_grad/tuple/control_dependency%^gradients/BiasAdd_2_grad/BiasAddGrad

1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentity0gradients/Conv2D_3_grad/tuple/control_dependency*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
đ
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad*
_output_shapes	
:
ą
9gradients/batch_normalization_2/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_2_grad/tuple/control_dependency"batch_normalization_2/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd

@gradients/batch_normalization_2/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_2/cond/Merge_grad/cond_grad
Í
Hgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_2/cond/Merge_grad/cond_gradA^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
Ń
Jgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_2/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_2/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
y
gradients/zeros_like_56	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:1*
_output_shapes	
:*
T0
y
gradients/zeros_like_57	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:2*
T0*
_output_shapes	
:
y
gradients/zeros_like_58	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_59	ZerosLike-batch_normalization_2/cond/FusedBatchNorm_1:4*
T0*
_output_shapes	
:

Mgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency2batch_normalization_2/cond/FusedBatchNorm_1/Switch4batch_normalization_2/cond/FusedBatchNorm_1/Switch_14batch_normalization_2/cond/FusedBatchNorm_1/Switch_34batch_normalization_2/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd::::*
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_60	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:1*
T0*
_output_shapes	
:
w
gradients/zeros_like_61	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_62	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_63	ZerosLike+batch_normalization_2/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:
ţ
Kgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_2/cond/Merge_grad/tuple/control_dependency_12batch_normalization_2/cond/FusedBatchNorm/Switch:14batch_normalization_2/cond/FusedBatchNorm/Switch_1:1+batch_normalization_2/cond/FusedBatchNorm:3+batch_normalization_2/cond/FusedBatchNorm:4*
data_formatNHWC*F
_output_shapes4
2:˙˙˙˙˙˙˙˙˙dd::: : *
is_training(*
epsilon%o:*
T0

Igradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ý
Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
˘
gradients/Switch_42SwitchConv2D_2"batch_normalization_2/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
g
gradients/Shape_43Shapegradients/Switch_42:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_42/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_42Fillgradients/Shape_43gradients/zeros_42/Const*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd*
T0*

index_type0

Kgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_42*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 

gradients/Switch_43Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_44Shapegradients/Switch_43:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_43/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_43Fillgradients/Shape_44gradients/zeros_43/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_43*
T0*
N*
_output_shapes
	:: 

gradients/Switch_44Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_45Shapegradients/Switch_44:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_44/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_44Fillgradients/Shape_45gradients/zeros_44/Const*
_output_shapes	
:*
T0*

index_type0
ň
Mgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_2/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_44*
N*
_output_shapes
	:: *
T0
˘
gradients/Switch_45SwitchConv2D_2"batch_normalization_2/cond/pred_id*
T0*L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙dd:˙˙˙˙˙˙˙˙˙dd
e
gradients/Shape_46Shapegradients/Switch_45*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_45/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_45Fillgradients/Shape_46gradients/zeros_45/Const*
T0*

index_type0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd
˙
Igradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_45Qgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙dd: 

gradients/Switch_46Switch batch_normalization_2/gamma/read"batch_normalization_2/cond/pred_id*"
_output_shapes
::*
T0
e
gradients/Shape_47Shapegradients/Switch_46*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_46/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_46Fillgradients/Shape_47gradients/zeros_46/Const*

index_type0*
_output_shapes	
:*
T0
î
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_46Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
N*
_output_shapes
	:: *
T0

gradients/Switch_47Switchbatch_normalization_2/beta/read"batch_normalization_2/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_48Shapegradients/Switch_47*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_47/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_47Fillgradients/Shape_48gradients/zeros_47/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_47Sgradients/batch_normalization_2/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
N*
_output_shapes
	:: *
T0
Ő
gradients/AddN_22AddNKgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙dd

gradients/Conv2D_2_grad/ShapeNShapeN	BiasAdd_1Variable_4/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_2_grad/ConstConst*%
valueB"            *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/readgradients/AddN_22*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0
˘
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	BiasAdd_1gradients/Conv2D_2_grad/Constgradients/AddN_22*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*(
_output_shapes
:*
	dilations
*
T0

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter

0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0

2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*(
_output_shapes
:
Ć
gradients/AddN_23AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_24AddNMgradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_2/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_2/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:
˘
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_2_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:*
T0

)gradients/BiasAdd_1_grad/tuple/group_depsNoOp1^gradients/Conv2D_2_grad/tuple/control_dependency%^gradients/BiasAdd_1_grad/BiasAddGrad

1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentity0gradients/Conv2D_2_grad/tuple/control_dependency*^gradients/BiasAdd_1_grad/tuple/group_deps*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
đ
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes	
:*
T0
ľ
9gradients/batch_normalization_1/cond/Merge_grad/cond_gradSwitch1gradients/BiasAdd_1_grad/tuple/control_dependency"batch_normalization_1/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ

@gradients/batch_normalization_1/cond/Merge_grad/tuple/group_depsNoOp:^gradients/batch_normalization_1/cond/Merge_grad/cond_grad
Ď
Hgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependencyIdentity9gradients/batch_normalization_1/cond/Merge_grad/cond_gradA^gradients/batch_normalization_1/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ
Ó
Jgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependency_1Identity;gradients/batch_normalization_1/cond/Merge_grad/cond_grad:1A^gradients/batch_normalization_1/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ
y
gradients/zeros_like_64	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:1*
_output_shapes	
:*
T0
y
gradients/zeros_like_65	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:2*
_output_shapes	
:*
T0
y
gradients/zeros_like_66	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:3*
T0*
_output_shapes	
:
y
gradients/zeros_like_67	ZerosLike-batch_normalization_1/cond/FusedBatchNorm_1:4*
_output_shapes	
:*
T0

Mgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependency2batch_normalization_1/cond/FusedBatchNorm_1/Switch4batch_normalization_1/cond/FusedBatchNorm_1/Switch_14batch_normalization_1/cond/FusedBatchNorm_1/Switch_34batch_normalization_1/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ::::*
is_training( *
epsilon%o:
Ł
Kgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpN^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Sgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGradL^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_deps*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Ugradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityOgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1L^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:

Ugradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityOgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2L^gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes	
:
w
gradients/zeros_like_68	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:1*
_output_shapes	
:*
T0
w
gradients/zeros_like_69	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
_output_shapes	
:
w
gradients/zeros_like_70	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:3*
T0*
_output_shapes	
:
w
gradients/zeros_like_71	ZerosLike+batch_normalization_1/cond/FusedBatchNorm:4*
T0*
_output_shapes	
:

Kgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradJgradients/batch_normalization_1/cond/Merge_grad/tuple/control_dependency_12batch_normalization_1/cond/FusedBatchNorm/Switch:14batch_normalization_1/cond/FusedBatchNorm/Switch_1:1+batch_normalization_1/cond/FusedBatchNorm:3+batch_normalization_1/cond/FusedBatchNorm:4*
epsilon%o:*
T0*
data_formatNHWC*H
_output_shapes6
4:˙˙˙˙˙˙˙˙˙ČČ::: : *
is_training(

Igradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_depsNoOpL^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Qgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityKgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGradJ^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:*
T0

Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes	
:
ý
Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
ý
Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityMgradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4J^gradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
Ś
gradients/Switch_48SwitchConv2D_1"batch_normalization_1/cond/pred_id*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ
g
gradients/Shape_49Shapegradients/Switch_48:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_48/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_48Fillgradients/Shape_49gradients/zeros_48/Const*
T0*

index_type0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

Kgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeSgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_48*
T0*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙ČČ: 

gradients/Switch_49Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_50Shapegradients/Switch_49:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_49/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_49Fillgradients/Shape_50gradients/zeros_49/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeUgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_49*
T0*
N*
_output_shapes
	:: 

gradients/Switch_50Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*"
_output_shapes
::
g
gradients/Shape_51Shapegradients/Switch_50:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_50/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_50Fillgradients/Shape_51gradients/zeros_50/Const*
T0*

index_type0*
_output_shapes	
:
ň
Mgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeUgradients/batch_normalization_1/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_50*
T0*
N*
_output_shapes
	:: 
Ś
gradients/Switch_51SwitchConv2D_1"batch_normalization_1/cond/pred_id*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙ČČ:˙˙˙˙˙˙˙˙˙ČČ
e
gradients/Shape_52Shapegradients/Switch_51*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_51/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_51Fillgradients/Shape_52gradients/zeros_51/Const*
T0*

index_type0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

Igradients/batch_normalization_1/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_51Qgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙ČČ: 

gradients/Switch_52Switch batch_normalization_1/gamma/read"batch_normalization_1/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_53Shapegradients/Switch_52*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_52/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_52Fillgradients/Shape_53gradients/zeros_52/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_52Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
	:: 

gradients/Switch_53Switchbatch_normalization_1/beta/read"batch_normalization_1/cond/pred_id*
T0*"
_output_shapes
::
e
gradients/Shape_54Shapegradients/Switch_53*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_53/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_53Fillgradients/Shape_54gradients/zeros_53/Const*
T0*

index_type0*
_output_shapes	
:
î
Kgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_53Sgradients/batch_normalization_1/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
T0*
N*
_output_shapes
	:: 
×
gradients/AddN_25AddNKgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_grad/cond_gradIgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ČČ

gradients/Conv2D_1_grad/ShapeNShapeNBiasAddVariable_2/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/Conv2D_1_grad/ConstConst*%
valueB"      @      *
dtype0*
_output_shapes
:
É
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/readgradients/AddN_25*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterBiasAddgradients/Conv2D_1_grad/Constgradients/AddN_25*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*'
_output_shapes
:@
Ć
gradients/AddN_26AddNMgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradKgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes	
:
Ć
gradients/AddN_27AddNMgradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradKgradients/batch_normalization_1/cond/FusedBatchNorm/Switch_2_grad/cond_grad*
T0*`
_classV
TRloc:@gradients/batch_normalization_1/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes	
:

"gradients/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients/Conv2D_1_grad/tuple/control_dependency*
_output_shapes
:@*
T0*
data_formatNHWC

'gradients/BiasAdd_grad/tuple/group_depsNoOp1^gradients/Conv2D_1_grad/tuple/control_dependency#^gradients/BiasAdd_grad/BiasAddGrad

/gradients/BiasAdd_grad/tuple/control_dependencyIdentity0gradients/Conv2D_1_grad/tuple/control_dependency(^gradients/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ç
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
­
7gradients/batch_normalization/cond/Merge_grad/cond_gradSwitch/gradients/BiasAdd_grad/tuple/control_dependency batch_normalization/cond/pred_id*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@

>gradients/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp8^gradients/batch_normalization/cond/Merge_grad/cond_grad
Č
Fgradients/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentity7gradients/batch_normalization/cond/Merge_grad/cond_grad?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
Ě
Hgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_1Identity9gradients/batch_normalization/cond/Merge_grad/cond_grad:1?^gradients/batch_normalization/cond/Merge_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
v
gradients/zeros_like_72	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:1*
_output_shapes
:@*
T0
v
gradients/zeros_like_73	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:2*
T0*
_output_shapes
:@
v
gradients/zeros_like_74	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:3*
T0*
_output_shapes
:@
v
gradients/zeros_like_75	ZerosLike+batch_normalization/cond/FusedBatchNorm_1:4*
_output_shapes
:@*
T0

Kgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradFgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_12batch_normalization/cond/FusedBatchNorm_1/Switch_32batch_normalization/cond/FusedBatchNorm_1/Switch_4*
T0*
data_formatNHWC*I
_output_shapes7
5:˙˙˙˙˙˙˙˙˙ČČ@:@:@:@:@*
is_training( *
epsilon%o:

Igradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOpL^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad

Qgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityKgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradJ^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
˙
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@
˙
Sgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityMgradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2J^gradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
t
gradients/zeros_like_76	ZerosLike)batch_normalization/cond/FusedBatchNorm:1*
T0*
_output_shapes
:@
t
gradients/zeros_like_77	ZerosLike)batch_normalization/cond/FusedBatchNorm:2*
T0*
_output_shapes
:@
t
gradients/zeros_like_78	ZerosLike)batch_normalization/cond/FusedBatchNorm:3*
T0*
_output_shapes
:@
t
gradients/zeros_like_79	ZerosLike)batch_normalization/cond/FusedBatchNorm:4*
_output_shapes
:@*
T0
ń
Igradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradHgradients/batch_normalization/cond/Merge_grad/tuple/control_dependency_10batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:1)batch_normalization/cond/FusedBatchNorm:3)batch_normalization/cond/FusedBatchNorm:4*
T0*
data_formatNHWC*E
_output_shapes3
1:˙˙˙˙˙˙˙˙˙ČČ@:@:@: : *
is_training(*
epsilon%o:

Ggradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOpJ^gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad

Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityIgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradH^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
T0
÷
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
÷
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
ő
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
_output_shapes
: *
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
ő
Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityKgradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4H^gradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
 
gradients/Switch_54SwitchConv2D batch_normalization/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0
g
gradients/Shape_55Shapegradients/Switch_54:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_54/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_54Fillgradients/Shape_55gradients/zeros_54/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@

Igradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMergeQgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencygradients/zeros_54*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 

gradients/Switch_55Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0* 
_output_shapes
:@:@
g
gradients/Shape_56Shapegradients/Switch_55:1*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_55/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_55Fillgradients/Shape_56gradients/zeros_55/Const*
_output_shapes
:@*
T0*

index_type0
í
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1gradients/zeros_55*
T0*
N*
_output_shapes

:@: 

gradients/Switch_56Switchbatch_normalization/beta/read batch_normalization/cond/pred_id*
T0* 
_output_shapes
:@:@
g
gradients/Shape_57Shapegradients/Switch_56:1*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_56/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_56Fillgradients/Shape_57gradients/zeros_56/Const*
T0*

index_type0*
_output_shapes
:@
í
Kgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMergeSgradients/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2gradients/zeros_56*
_output_shapes

:@: *
T0*
N
 
gradients/Switch_57SwitchConv2D batch_normalization/cond/pred_id*N
_output_shapes<
::˙˙˙˙˙˙˙˙˙ČČ@:˙˙˙˙˙˙˙˙˙ČČ@*
T0
e
gradients/Shape_58Shapegradients/Switch_57*
T0*
out_type0*
_output_shapes
:
]
gradients/zeros_57/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_57Fillgradients/Shape_58gradients/zeros_57/Const*
T0*

index_type0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@
ü
Ggradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergegradients/zeros_57Ogradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency*
T0*
N*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙ČČ@: 

gradients/Switch_58Switchbatch_normalization/gamma/read batch_normalization/cond/pred_id*
T0* 
_output_shapes
:@:@
e
gradients/Shape_59Shapegradients/Switch_58*
_output_shapes
:*
T0*
out_type0
]
gradients/zeros_58/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

gradients/zeros_58Fillgradients/Shape_59gradients/zeros_58/Const*
T0*

index_type0*
_output_shapes
:@
é
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMergegradients/zeros_58Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes

:@: 

gradients/Switch_59Switchbatch_normalization/beta/read batch_normalization/cond/pred_id* 
_output_shapes
:@:@*
T0
e
gradients/Shape_60Shapegradients/Switch_59*
out_type0*
_output_shapes
:*
T0
]
gradients/zeros_59/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

gradients/zeros_59Fillgradients/Shape_60gradients/zeros_59/Const*
_output_shapes
:@*
T0*

index_type0
é
Igradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMergegradients/zeros_59Qgradients/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2*
_output_shapes

:@: *
T0*
N
Đ
gradients/AddN_28AddNIgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradGgradients/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad*
N*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ČČ@*
T0*\
_classR
PNloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad
|
gradients/Conv2D_grad/ShapeNShapeNxVariable/read*
T0*
out_type0*
N* 
_output_shapes
::
t
gradients/Conv2D_grad/ConstConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ă
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/readgradients/AddN_28*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
	dilations
*
T0

*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterxgradients/Conv2D_grad/Constgradients/AddN_28*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:@
ż
gradients/AddN_29AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad*
T0*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:@
ż
gradients/AddN_30AddNKgradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradIgradients/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad*^
_classT
RPloc:@gradients/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:@*
T0
Š
3Variable/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"         @   *
_class
loc:@Variable*
dtype0*
_output_shapes
:

)Variable/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable*
dtype0*
_output_shapes
: 
ë
#Variable/Adadelta/Initializer/zerosFill3Variable/Adadelta/Initializer/zeros/shape_as_tensor)Variable/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable*&
_output_shapes
:@
˛
Variable/Adadelta
VariableV2*
	container *
shape:@*
dtype0*&
_output_shapes
:@*
shared_name *
_class
loc:@Variable
Ń
Variable/Adadelta/AssignAssignVariable/Adadelta#Variable/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:@

Variable/Adadelta/readIdentityVariable/Adadelta*
T0*
_class
loc:@Variable*&
_output_shapes
:@
Ť
5Variable/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         @   *
_class
loc:@Variable*
dtype0*
_output_shapes
:

+Variable/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable*
dtype0*
_output_shapes
: 
ń
%Variable/Adadelta_1/Initializer/zerosFill5Variable/Adadelta_1/Initializer/zeros/shape_as_tensor+Variable/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable*&
_output_shapes
:@
´
Variable/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape:@*
dtype0*&
_output_shapes
:@
×
Variable/Adadelta_1/AssignAssignVariable/Adadelta_1%Variable/Adadelta_1/Initializer/zeros*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0

Variable/Adadelta_1/readIdentityVariable/Adadelta_1*
T0*
_class
loc:@Variable*&
_output_shapes
:@

5Variable_1/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@Variable_1*
dtype0*
_output_shapes
:

+Variable_1/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
ç
%Variable_1/Adadelta/Initializer/zerosFill5Variable_1/Adadelta/Initializer/zeros/shape_as_tensor+Variable_1/Adadelta/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*
_class
loc:@Variable_1

Variable_1/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:@*
dtype0*
_output_shapes
:@
Í
Variable_1/Adadelta/AssignAssignVariable_1/Adadelta%Variable_1/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:@
}
Variable_1/Adadelta/readIdentityVariable_1/Adadelta*
T0*
_class
loc:@Variable_1*
_output_shapes
:@
 
7Variable_1/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:@*
_class
loc:@Variable_1*
dtype0

-Variable_1/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
í
'Variable_1/Adadelta_1/Initializer/zerosFill7Variable_1/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_1/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_1*
_output_shapes
:@
 
Variable_1/Adadelta_1
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@Variable_1
Ó
Variable_1/Adadelta_1/AssignAssignVariable_1/Adadelta_1'Variable_1/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes
:@

Variable_1/Adadelta_1/readIdentityVariable_1/Adadelta_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:@
ź
Dbatch_normalization/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:@*,
_class"
 loc:@batch_normalization/gamma*
dtype0
­
:batch_normalization/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
: 
Ł
4batch_normalization/gamma/Adadelta/Initializer/zerosFillDbatch_normalization/gamma/Adadelta/Initializer/zeros/shape_as_tensor:batch_normalization/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
ź
"batch_normalization/gamma/Adadelta
VariableV2*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@

)batch_normalization/gamma/Adadelta/AssignAssign"batch_normalization/gamma/Adadelta4batch_normalization/gamma/Adadelta/Initializer/zeros*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ş
'batch_normalization/gamma/Adadelta/readIdentity"batch_normalization/gamma/Adadelta*
_output_shapes
:@*
T0*,
_class"
 loc:@batch_normalization/gamma
ž
Fbatch_normalization/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:@*,
_class"
 loc:@batch_normalization/gamma
Ż
<batch_normalization/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
: 
Š
6batch_normalization/gamma/Adadelta_1/Initializer/zerosFillFbatch_normalization/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor<batch_normalization/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
ž
$batch_normalization/gamma/Adadelta_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container 

+batch_normalization/gamma/Adadelta_1/AssignAssign$batch_normalization/gamma/Adadelta_16batch_normalization/gamma/Adadelta_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(
Ž
)batch_normalization/gamma/Adadelta_1/readIdentity$batch_normalization/gamma/Adadelta_1*
_output_shapes
:@*
T0*,
_class"
 loc:@batch_normalization/gamma
ş
Cbatch_normalization/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
Ť
9batch_normalization/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
: 

3batch_normalization/beta/Adadelta/Initializer/zerosFillCbatch_normalization/beta/Adadelta/Initializer/zeros/shape_as_tensor9batch_normalization/beta/Adadelta/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*+
_class!
loc:@batch_normalization/beta
ş
!batch_normalization/beta/Adadelta
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container 

(batch_normalization/beta/Adadelta/AssignAssign!batch_normalization/beta/Adadelta3batch_normalization/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@
§
&batch_normalization/beta/Adadelta/readIdentity!batch_normalization/beta/Adadelta*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ź
Ebatch_normalization/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
­
;batch_normalization/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
: 
Ľ
5batch_normalization/beta/Adadelta_1/Initializer/zerosFillEbatch_normalization/beta/Adadelta_1/Initializer/zeros/shape_as_tensor;batch_normalization/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ź
#batch_normalization/beta/Adadelta_1
VariableV2*+
_class!
loc:@batch_normalization/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 

*batch_normalization/beta/Adadelta_1/AssignAssign#batch_normalization/beta/Adadelta_15batch_normalization/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@
Ť
(batch_normalization/beta/Adadelta_1/readIdentity#batch_normalization/beta/Adadelta_1*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
­
5Variable_2/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"      @      *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:

+Variable_2/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
ô
%Variable_2/Adadelta/Initializer/zerosFill5Variable_2/Adadelta/Initializer/zeros/shape_as_tensor+Variable_2/Adadelta/Initializer/zeros/Const*'
_output_shapes
:@*
T0*

index_type0*
_class
loc:@Variable_2
¸
Variable_2/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape:@*
dtype0*'
_output_shapes
:@
Ú
Variable_2/Adadelta/AssignAssignVariable_2/Adadelta%Variable_2/Adadelta/Initializer/zeros*'
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(

Variable_2/Adadelta/readIdentityVariable_2/Adadelta*
_class
loc:@Variable_2*'
_output_shapes
:@*
T0
Ż
7Variable_2/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @      *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:

-Variable_2/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
ú
'Variable_2/Adadelta_1/Initializer/zerosFill7Variable_2/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_2/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*'
_output_shapes
:@
ş
Variable_2/Adadelta_1
VariableV2*
dtype0*'
_output_shapes
:@*
shared_name *
_class
loc:@Variable_2*
	container *
shape:@
ŕ
Variable_2/Adadelta_1/AssignAssignVariable_2/Adadelta_1'Variable_2/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*'
_output_shapes
:@

Variable_2/Adadelta_1/readIdentityVariable_2/Adadelta_1*
T0*
_class
loc:@Variable_2*'
_output_shapes
:@

5Variable_3/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

+Variable_3/Adadelta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_3
č
%Variable_3/Adadelta/Initializer/zerosFill5Variable_3/Adadelta/Initializer/zeros/shape_as_tensor+Variable_3/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_3*
_output_shapes	
:
 
Variable_3/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_3*
	container *
shape:
Î
Variable_3/Adadelta/AssignAssignVariable_3/Adadelta%Variable_3/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:
~
Variable_3/Adadelta/readIdentityVariable_3/Adadelta*
T0*
_class
loc:@Variable_3*
_output_shapes	
:
Ą
7Variable_3/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

-Variable_3/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
î
'Variable_3/Adadelta_1/Initializer/zerosFill7Variable_3/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_3/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*
_class
loc:@Variable_3
˘
Variable_3/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_3*
	container *
shape:
Ô
Variable_3/Adadelta_1/AssignAssignVariable_3/Adadelta_1'Variable_3/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes	
:

Variable_3/Adadelta_1/readIdentityVariable_3/Adadelta_1*
T0*
_class
loc:@Variable_3*
_output_shapes	
:
Á
Fbatch_normalization_1/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_1/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_1/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_1/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_1/gamma/Adadelta/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma
Â
$batch_normalization_1/gamma/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container 

+batch_normalization_1/gamma/Adadelta/AssignAssign$batch_normalization_1/gamma/Adadelta6batch_normalization_1/gamma/Adadelta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma
ą
)batch_normalization_1/gamma/Adadelta/readIdentity$batch_normalization_1/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_1/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_1/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_1/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_1/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_1/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:
Ä
&batch_normalization_1/gamma/Adadelta_1
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

-batch_normalization_1/gamma/Adadelta_1/AssignAssign&batch_normalization_1/gamma/Adadelta_18batch_normalization_1/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_1/gamma/Adadelta_1/readIdentity&batch_normalization_1/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:
ż
Ebatch_normalization_1/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*-
_class#
!loc:@batch_normalization_1/beta
Ż
;batch_normalization_1/beta/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0
¨
5batch_normalization_1/beta/Adadelta/Initializer/zerosFillEbatch_normalization_1/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_1/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
Ŕ
#batch_normalization_1/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

*batch_normalization_1/beta/Adadelta/AssignAssign#batch_normalization_1/beta/Adadelta5batch_normalization_1/beta/Adadelta/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(
Ž
(batch_normalization_1/beta/Adadelta/readIdentity#batch_normalization_1/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
Á
Gbatch_normalization_1/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_1/beta/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_1/beta
Ž
7batch_normalization_1/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_1/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_1/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
Â
%batch_normalization_1/beta/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container 

,batch_normalization_1/beta/Adadelta_1/AssignAssign%batch_normalization_1/beta/Adadelta_17batch_normalization_1/beta/Adadelta_1/Initializer/zeros*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˛
*batch_normalization_1/beta/Adadelta_1/readIdentity%batch_normalization_1/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
­
5Variable_4/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_4*
dtype0*
_output_shapes
:

+Variable_4/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_4*
dtype0
ő
%Variable_4/Adadelta/Initializer/zerosFill5Variable_4/Adadelta/Initializer/zeros/shape_as_tensor+Variable_4/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_4*(
_output_shapes
:
ş
Variable_4/Adadelta
VariableV2*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_4*
	container *
shape:
Ű
Variable_4/Adadelta/AssignAssignVariable_4/Adadelta%Variable_4/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*(
_output_shapes
:

Variable_4/Adadelta/readIdentityVariable_4/Adadelta*
T0*
_class
loc:@Variable_4*(
_output_shapes
:
Ż
7Variable_4/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_4*
dtype0*
_output_shapes
:

-Variable_4/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
ű
'Variable_4/Adadelta_1/Initializer/zerosFill7Variable_4/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_4/Adadelta_1/Initializer/zeros/Const*(
_output_shapes
:*
T0*

index_type0*
_class
loc:@Variable_4
ź
Variable_4/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
	container *
shape:*
dtype0*(
_output_shapes
:
á
Variable_4/Adadelta_1/AssignAssignVariable_4/Adadelta_1'Variable_4/Adadelta_1/Initializer/zeros*(
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(

Variable_4/Adadelta_1/readIdentityVariable_4/Adadelta_1*
T0*
_class
loc:@Variable_4*(
_output_shapes
:

5Variable_5/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

+Variable_5/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
č
%Variable_5/Adadelta/Initializer/zerosFill5Variable_5/Adadelta/Initializer/zeros/shape_as_tensor+Variable_5/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_5*
_output_shapes	
:
 
Variable_5/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_5*
	container *
shape:
Î
Variable_5/Adadelta/AssignAssignVariable_5/Adadelta%Variable_5/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:
~
Variable_5/Adadelta/readIdentityVariable_5/Adadelta*
T0*
_class
loc:@Variable_5*
_output_shapes	
:
Ą
7Variable_5/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

-Variable_5/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
î
'Variable_5/Adadelta_1/Initializer/zerosFill7Variable_5/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_5/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_5*
_output_shapes	
:
˘
Variable_5/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_5*
	container *
shape:*
dtype0*
_output_shapes	
:
Ô
Variable_5/Adadelta_1/AssignAssignVariable_5/Adadelta_1'Variable_5/Adadelta_1/Initializer/zeros*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

Variable_5/Adadelta_1/readIdentityVariable_5/Adadelta_1*
T0*
_class
loc:@Variable_5*
_output_shapes	
:
Á
Fbatch_normalization_2/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_2/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_2/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_2/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_2/gamma/Adadelta/Initializer/zeros/Const*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:*
T0
Â
$batch_normalization_2/gamma/Adadelta
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

+batch_normalization_2/gamma/Adadelta/AssignAssign$batch_normalization_2/gamma/Adadelta6batch_normalization_2/gamma/Adadelta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma
ą
)batch_normalization_2/gamma/Adadelta/readIdentity$batch_normalization_2/gamma/Adadelta*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_2/gamma
Ă
Hbatch_normalization_2/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_2/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_2/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_2/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_2/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_2/gamma
Ä
&batch_normalization_2/gamma/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container 

-batch_normalization_2/gamma/Adadelta_1/AssignAssign&batch_normalization_2/gamma/Adadelta_18batch_normalization_2/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_2/gamma/Adadelta_1/readIdentity&batch_normalization_2/gamma/Adadelta_1*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_2/gamma
ż
Ebatch_normalization_2/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_2/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_2/beta/Adadelta/Initializer/zerosFillEbatch_normalization_2/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_2/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
Ŕ
#batch_normalization_2/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

*batch_normalization_2/beta/Adadelta/AssignAssign#batch_normalization_2/beta/Adadelta5batch_normalization_2/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_2/beta/Adadelta/readIdentity#batch_normalization_2/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
Á
Gbatch_normalization_2/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_2/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_2/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_2/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_2/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
Â
%batch_normalization_2/beta/Adadelta_1
VariableV2*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0

,batch_normalization_2/beta/Adadelta_1/AssignAssign%batch_normalization_2/beta/Adadelta_17batch_normalization_2/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_2/beta/Adadelta_1/readIdentity%batch_normalization_2/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
­
5Variable_6/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_6*
dtype0*
_output_shapes
:

+Variable_6/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
: 
ő
%Variable_6/Adadelta/Initializer/zerosFill5Variable_6/Adadelta/Initializer/zeros/shape_as_tensor+Variable_6/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_6*(
_output_shapes
:
ş
Variable_6/Adadelta
VariableV2*
_class
loc:@Variable_6*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
Ű
Variable_6/Adadelta/AssignAssignVariable_6/Adadelta%Variable_6/Adadelta/Initializer/zeros*
T0*
_class
loc:@Variable_6*
validate_shape(*(
_output_shapes
:*
use_locking(

Variable_6/Adadelta/readIdentityVariable_6/Adadelta*
T0*
_class
loc:@Variable_6*(
_output_shapes
:
Ż
7Variable_6/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_6*
dtype0*
_output_shapes
:

-Variable_6/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_6
ű
'Variable_6/Adadelta_1/Initializer/zerosFill7Variable_6/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_6/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_6*(
_output_shapes
:
ź
Variable_6/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape:*
dtype0*(
_output_shapes
:
á
Variable_6/Adadelta_1/AssignAssignVariable_6/Adadelta_1'Variable_6/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*(
_output_shapes
:

Variable_6/Adadelta_1/readIdentityVariable_6/Adadelta_1*
T0*
_class
loc:@Variable_6*(
_output_shapes
:

5Variable_7/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

+Variable_7/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
č
%Variable_7/Adadelta/Initializer/zerosFill5Variable_7/Adadelta/Initializer/zeros/shape_as_tensor+Variable_7/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_7*
_output_shapes	
:
 
Variable_7/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes	
:
Î
Variable_7/Adadelta/AssignAssignVariable_7/Adadelta%Variable_7/Adadelta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_7
~
Variable_7/Adadelta/readIdentityVariable_7/Adadelta*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
Ą
7Variable_7/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

-Variable_7/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_7*
dtype0
î
'Variable_7/Adadelta_1/Initializer/zerosFill7Variable_7/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_7/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_7*
_output_shapes	
:
˘
Variable_7/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:*
dtype0*
_output_shapes	
:
Ô
Variable_7/Adadelta_1/AssignAssignVariable_7/Adadelta_1'Variable_7/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes	
:

Variable_7/Adadelta_1/readIdentityVariable_7/Adadelta_1*
T0*
_class
loc:@Variable_7*
_output_shapes	
:
Á
Fbatch_normalization_3/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*.
_class$
" loc:@batch_normalization_3/gamma
ą
<batch_normalization_3/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_3/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_3/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_3/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
Â
$batch_normalization_3/gamma/Adadelta
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

+batch_normalization_3/gamma/Adadelta/AssignAssign$batch_normalization_3/gamma/Adadelta6batch_normalization_3/gamma/Adadelta/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ą
)batch_normalization_3/gamma/Adadelta/readIdentity$batch_normalization_3/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_3/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_3/gamma/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_3/gamma
˛
8batch_normalization_3/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_3/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_3/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
Ä
&batch_normalization_3/gamma/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container 

-batch_normalization_3/gamma/Adadelta_1/AssignAssign&batch_normalization_3/gamma/Adadelta_18batch_normalization_3/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_3/gamma/Adadelta_1/readIdentity&batch_normalization_3/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:
ż
Ebatch_normalization_3/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_3/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_3/beta/Adadelta/Initializer/zerosFillEbatch_normalization_3/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_3/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
Ŕ
#batch_normalization_3/beta/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:

*batch_normalization_3/beta/Adadelta/AssignAssign#batch_normalization_3/beta/Adadelta5batch_normalization_3/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_3/beta/Adadelta/readIdentity#batch_normalization_3/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
Á
Gbatch_normalization_3/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_3/beta/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_3/beta
Ž
7batch_normalization_3/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_3/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_3/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:
Â
%batch_normalization_3/beta/Adadelta_1
VariableV2*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:*
dtype0

,batch_normalization_3/beta/Adadelta_1/AssignAssign%batch_normalization_3/beta/Adadelta_17batch_normalization_3/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_3/beta/Adadelta_1/readIdentity%batch_normalization_3/beta/Adadelta_1*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_3/beta
­
5Variable_8/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:

+Variable_8/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_8*
dtype0*
_output_shapes
: 
ő
%Variable_8/Adadelta/Initializer/zerosFill5Variable_8/Adadelta/Initializer/zeros/shape_as_tensor+Variable_8/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_8*(
_output_shapes
:
ş
Variable_8/Adadelta
VariableV2*
shape:*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_8*
	container 
Ű
Variable_8/Adadelta/AssignAssignVariable_8/Adadelta%Variable_8/Adadelta/Initializer/zeros*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

Variable_8/Adadelta/readIdentityVariable_8/Adadelta*
T0*
_class
loc:@Variable_8*(
_output_shapes
:
Ż
7Variable_8/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"            *
_class
loc:@Variable_8*
dtype0

-Variable_8/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_8*
dtype0*
_output_shapes
: 
ű
'Variable_8/Adadelta_1/Initializer/zerosFill7Variable_8/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_8/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_8*(
_output_shapes
:
ź
Variable_8/Adadelta_1
VariableV2*
shape:*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_8*
	container 
á
Variable_8/Adadelta_1/AssignAssignVariable_8/Adadelta_1'Variable_8/Adadelta_1/Initializer/zeros*
T0*
_class
loc:@Variable_8*
validate_shape(*(
_output_shapes
:*
use_locking(

Variable_8/Adadelta_1/readIdentityVariable_8/Adadelta_1*
T0*
_class
loc:@Variable_8*(
_output_shapes
:

5Variable_9/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*
_class
loc:@Variable_9

+Variable_9/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
: 
č
%Variable_9/Adadelta/Initializer/zerosFill5Variable_9/Adadelta/Initializer/zeros/shape_as_tensor+Variable_9/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_9*
_output_shapes	
:
 
Variable_9/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_9*
	container *
shape:*
dtype0*
_output_shapes	
:
Î
Variable_9/Adadelta/AssignAssignVariable_9/Adadelta%Variable_9/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*
_output_shapes	
:
~
Variable_9/Adadelta/readIdentityVariable_9/Adadelta*
T0*
_class
loc:@Variable_9*
_output_shapes	
:
Ą
7Variable_9/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_9*
dtype0*
_output_shapes
:

-Variable_9/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
: 
î
'Variable_9/Adadelta_1/Initializer/zerosFill7Variable_9/Adadelta_1/Initializer/zeros/shape_as_tensor-Variable_9/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*
_class
loc:@Variable_9
˘
Variable_9/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_9*
	container *
shape:*
dtype0*
_output_shapes	
:
Ô
Variable_9/Adadelta_1/AssignAssignVariable_9/Adadelta_1'Variable_9/Adadelta_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(

Variable_9/Adadelta_1/readIdentityVariable_9/Adadelta_1*
T0*
_class
loc:@Variable_9*
_output_shapes	
:
Á
Fbatch_normalization_4/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*.
_class$
" loc:@batch_normalization_4/gamma
ą
<batch_normalization_4/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_4/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_4/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_4/gamma/Adadelta/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma
Â
$batch_normalization_4/gamma/Adadelta
VariableV2*.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

+batch_normalization_4/gamma/Adadelta/AssignAssign$batch_normalization_4/gamma/Adadelta6batch_normalization_4/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_4/gamma/Adadelta/readIdentity$batch_normalization_4/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_4/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_4/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_4/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_4/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_4/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
Ä
&batch_normalization_4/gamma/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_4/gamma*
	container 

-batch_normalization_4/gamma/Adadelta_1/AssignAssign&batch_normalization_4/gamma/Adadelta_18batch_normalization_4/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_4/gamma/Adadelta_1/readIdentity&batch_normalization_4/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
ż
Ebatch_normalization_4/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_4/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_4/beta/Adadelta/Initializer/zerosFillEbatch_normalization_4/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_4/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
Ŕ
#batch_normalization_4/beta/Adadelta
VariableV2*-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

*batch_normalization_4/beta/Adadelta/AssignAssign#batch_normalization_4/beta/Adadelta5batch_normalization_4/beta/Adadelta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta
Ž
(batch_normalization_4/beta/Adadelta/readIdentity#batch_normalization_4/beta/Adadelta*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_4/beta
Á
Gbatch_normalization_4/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_4/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_4/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_4/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_4/beta/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*-
_class#
!loc:@batch_normalization_4/beta
Â
%batch_normalization_4/beta/Adadelta_1
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

,batch_normalization_4/beta/Adadelta_1/AssignAssign%batch_normalization_4/beta/Adadelta_17batch_normalization_4/beta/Adadelta_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_4/beta*
validate_shape(
˛
*batch_normalization_4/beta/Adadelta_1/readIdentity%batch_normalization_4/beta/Adadelta_1*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:*
T0
Ż
6Variable_10/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

,Variable_10/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_10*
dtype0*
_output_shapes
: 
ů
&Variable_10/Adadelta/Initializer/zerosFill6Variable_10/Adadelta/Initializer/zeros/shape_as_tensor,Variable_10/Adadelta/Initializer/zeros/Const*(
_output_shapes
:*
T0*

index_type0*
_class
loc:@Variable_10
ź
Variable_10/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_10*
	container *
shape:*
dtype0*(
_output_shapes
:
ß
Variable_10/Adadelta/AssignAssignVariable_10/Adadelta&Variable_10/Adadelta/Initializer/zeros*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_10

Variable_10/Adadelta/readIdentityVariable_10/Adadelta*
_class
loc:@Variable_10*(
_output_shapes
:*
T0
ą
8Variable_10/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

.Variable_10/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_10*
dtype0*
_output_shapes
: 
˙
(Variable_10/Adadelta_1/Initializer/zerosFill8Variable_10/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_10/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_10*(
_output_shapes
:
ž
Variable_10/Adadelta_1
VariableV2*
_class
loc:@Variable_10*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
ĺ
Variable_10/Adadelta_1/AssignAssignVariable_10/Adadelta_1(Variable_10/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*(
_output_shapes
:

Variable_10/Adadelta_1/readIdentityVariable_10/Adadelta_1*
T0*
_class
loc:@Variable_10*(
_output_shapes
:
Ą
6Variable_11/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

,Variable_11/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_11*
dtype0*
_output_shapes
: 
ě
&Variable_11/Adadelta/Initializer/zerosFill6Variable_11/Adadelta/Initializer/zeros/shape_as_tensor,Variable_11/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_11*
_output_shapes	
:
˘
Variable_11/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_11*
	container 
Ň
Variable_11/Adadelta/AssignAssignVariable_11/Adadelta&Variable_11/Adadelta/Initializer/zeros*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:*
use_locking(

Variable_11/Adadelta/readIdentityVariable_11/Adadelta*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
Ł
8Variable_11/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*
_class
loc:@Variable_11

.Variable_11/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_11*
dtype0
ň
(Variable_11/Adadelta_1/Initializer/zerosFill8Variable_11/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_11/Adadelta_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_11*
_output_shapes	
:*
T0
¤
Variable_11/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_11*
	container 
Ř
Variable_11/Adadelta_1/AssignAssignVariable_11/Adadelta_1(Variable_11/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*
_output_shapes	
:

Variable_11/Adadelta_1/readIdentityVariable_11/Adadelta_1*
T0*
_class
loc:@Variable_11*
_output_shapes	
:
Á
Fbatch_normalization_5/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_5/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_5/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_5/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_5/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
Â
$batch_normalization_5/gamma/Adadelta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma

+batch_normalization_5/gamma/Adadelta/AssignAssign$batch_normalization_5/gamma/Adadelta6batch_normalization_5/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_5/gamma/Adadelta/readIdentity$batch_normalization_5/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_5/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_5/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_5/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_5/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_5/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
Ä
&batch_normalization_5/gamma/Adadelta_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_5/gamma

-batch_normalization_5/gamma/Adadelta_1/AssignAssign&batch_normalization_5/gamma/Adadelta_18batch_normalization_5/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_5/gamma/Adadelta_1/readIdentity&batch_normalization_5/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes	
:
ż
Ebatch_normalization_5/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_5/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_5/beta/Adadelta/Initializer/zerosFillEbatch_normalization_5/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_5/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Ŕ
#batch_normalization_5/beta/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_5/beta*
	container 

*batch_normalization_5/beta/Adadelta/AssignAssign#batch_normalization_5/beta/Adadelta5batch_normalization_5/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_5/beta/Adadelta/readIdentity#batch_normalization_5/beta/Adadelta*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_5/beta
Á
Gbatch_normalization_5/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_5/beta/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_5/beta
Ž
7batch_normalization_5/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_5/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_5/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Â
%batch_normalization_5/beta/Adadelta_1
VariableV2*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_5/beta*
	container *
shape:*
dtype0

,batch_normalization_5/beta/Adadelta_1/AssignAssign%batch_normalization_5/beta/Adadelta_17batch_normalization_5/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_5/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_5/beta/Adadelta_1/readIdentity%batch_normalization_5/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
Ż
6Variable_12/Adadelta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"            *
_class
loc:@Variable_12*
dtype0

,Variable_12/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
ů
&Variable_12/Adadelta/Initializer/zerosFill6Variable_12/Adadelta/Initializer/zeros/shape_as_tensor,Variable_12/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_12*(
_output_shapes
:
ź
Variable_12/Adadelta
VariableV2*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_12*
	container *
shape:
ß
Variable_12/Adadelta/AssignAssignVariable_12/Adadelta&Variable_12/Adadelta/Initializer/zeros*
T0*
_class
loc:@Variable_12*
validate_shape(*(
_output_shapes
:*
use_locking(

Variable_12/Adadelta/readIdentityVariable_12/Adadelta*(
_output_shapes
:*
T0*
_class
loc:@Variable_12
ą
8Variable_12/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            *
_class
loc:@Variable_12

.Variable_12/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
˙
(Variable_12/Adadelta_1/Initializer/zerosFill8Variable_12/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_12/Adadelta_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_12*(
_output_shapes
:*
T0
ž
Variable_12/Adadelta_1
VariableV2*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_12*
	container *
shape:*
dtype0
ĺ
Variable_12/Adadelta_1/AssignAssignVariable_12/Adadelta_1(Variable_12/Adadelta_1/Initializer/zeros*
_class
loc:@Variable_12*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

Variable_12/Adadelta_1/readIdentityVariable_12/Adadelta_1*
T0*
_class
loc:@Variable_12*(
_output_shapes
:
Ą
6Variable_13/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_13*
dtype0*
_output_shapes
:

,Variable_13/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
: 
ě
&Variable_13/Adadelta/Initializer/zerosFill6Variable_13/Adadelta/Initializer/zeros/shape_as_tensor,Variable_13/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_13*
_output_shapes	
:
˘
Variable_13/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_13*
	container *
shape:
Ň
Variable_13/Adadelta/AssignAssignVariable_13/Adadelta&Variable_13/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*
_output_shapes	
:

Variable_13/Adadelta/readIdentityVariable_13/Adadelta*
_output_shapes	
:*
T0*
_class
loc:@Variable_13
Ł
8Variable_13/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_13*
dtype0*
_output_shapes
:

.Variable_13/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
: 
ň
(Variable_13/Adadelta_1/Initializer/zerosFill8Variable_13/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_13/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*
_class
loc:@Variable_13
¤
Variable_13/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@Variable_13*
	container 
Ř
Variable_13/Adadelta_1/AssignAssignVariable_13/Adadelta_1(Variable_13/Adadelta_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(

Variable_13/Adadelta_1/readIdentityVariable_13/Adadelta_1*
T0*
_class
loc:@Variable_13*
_output_shapes	
:
Á
Fbatch_normalization_6/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_6/gamma/Adadelta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_6/gamma
Ź
6batch_normalization_6/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_6/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_6/gamma/Adadelta/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
Â
$batch_normalization_6/gamma/Adadelta
VariableV2*.
_class$
" loc:@batch_normalization_6/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

+batch_normalization_6/gamma/Adadelta/AssignAssign$batch_normalization_6/gamma/Adadelta6batch_normalization_6/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_6/gamma/Adadelta/readIdentity$batch_normalization_6/gamma/Adadelta*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_6/gamma
Ă
Hbatch_normalization_6/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_6/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_6/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_6/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_6/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_6/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
Ä
&batch_normalization_6/gamma/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_6/gamma*
	container 

-batch_normalization_6/gamma/Adadelta_1/AssignAssign&batch_normalization_6/gamma/Adadelta_18batch_normalization_6/gamma/Adadelta_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:
ľ
+batch_normalization_6/gamma/Adadelta_1/readIdentity&batch_normalization_6/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes	
:
ż
Ebatch_normalization_6/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_6/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_6/beta/Adadelta/Initializer/zerosFillEbatch_normalization_6/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_6/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
Ŕ
#batch_normalization_6/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_6/beta*
	container *
shape:*
dtype0*
_output_shapes	
:

*batch_normalization_6/beta/Adadelta/AssignAssign#batch_normalization_6/beta/Adadelta5batch_normalization_6/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_6/beta/Adadelta/readIdentity#batch_normalization_6/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
Á
Gbatch_normalization_6/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_6/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_6/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_6/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_6/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_6/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
Â
%batch_normalization_6/beta/Adadelta_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_6/beta*
	container *
shape:

,batch_normalization_6/beta/Adadelta_1/AssignAssign%batch_normalization_6/beta/Adadelta_17batch_normalization_6/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:
˛
*batch_normalization_6/beta/Adadelta_1/readIdentity%batch_normalization_6/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
Ż
6Variable_14/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"            *
_class
loc:@Variable_14*
dtype0*
_output_shapes
:

,Variable_14/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_14*
dtype0
ů
&Variable_14/Adadelta/Initializer/zerosFill6Variable_14/Adadelta/Initializer/zeros/shape_as_tensor,Variable_14/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_14*(
_output_shapes
:
ź
Variable_14/Adadelta
VariableV2*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_14*
	container *
shape:
ß
Variable_14/Adadelta/AssignAssignVariable_14/Adadelta&Variable_14/Adadelta/Initializer/zeros*
T0*
_class
loc:@Variable_14*
validate_shape(*(
_output_shapes
:*
use_locking(

Variable_14/Adadelta/readIdentityVariable_14/Adadelta*
T0*
_class
loc:@Variable_14*(
_output_shapes
:
ą
8Variable_14/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"            *
_class
loc:@Variable_14

.Variable_14/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_14*
dtype0*
_output_shapes
: 
˙
(Variable_14/Adadelta_1/Initializer/zerosFill8Variable_14/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_14/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_14*(
_output_shapes
:
ž
Variable_14/Adadelta_1
VariableV2*
shape:*
dtype0*(
_output_shapes
:*
shared_name *
_class
loc:@Variable_14*
	container 
ĺ
Variable_14/Adadelta_1/AssignAssignVariable_14/Adadelta_1(Variable_14/Adadelta_1/Initializer/zeros*
_class
loc:@Variable_14*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0

Variable_14/Adadelta_1/readIdentityVariable_14/Adadelta_1*(
_output_shapes
:*
T0*
_class
loc:@Variable_14
Ą
6Variable_15/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

,Variable_15/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
: 
ě
&Variable_15/Adadelta/Initializer/zerosFill6Variable_15/Adadelta/Initializer/zeros/shape_as_tensor,Variable_15/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_15*
_output_shapes	
:
˘
Variable_15/Adadelta
VariableV2*
_class
loc:@Variable_15*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ň
Variable_15/Adadelta/AssignAssignVariable_15/Adadelta&Variable_15/Adadelta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@Variable_15

Variable_15/Adadelta/readIdentityVariable_15/Adadelta*
T0*
_class
loc:@Variable_15*
_output_shapes	
:
Ł
8Variable_15/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

.Variable_15/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
: 
ň
(Variable_15/Adadelta_1/Initializer/zerosFill8Variable_15/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_15/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_15*
_output_shapes	
:
¤
Variable_15/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
Variable_15/Adadelta_1/AssignAssignVariable_15/Adadelta_1(Variable_15/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes	
:

Variable_15/Adadelta_1/readIdentityVariable_15/Adadelta_1*
T0*
_class
loc:@Variable_15*
_output_shapes	
:
Á
Fbatch_normalization_7/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_7/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
: 
Ź
6batch_normalization_7/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_7/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_7/gamma/Adadelta/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_7/gamma
Â
$batch_normalization_7/gamma/Adadelta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *.
_class$
" loc:@batch_normalization_7/gamma*
	container *
shape:

+batch_normalization_7/gamma/Adadelta/AssignAssign$batch_normalization_7/gamma/Adadelta6batch_normalization_7/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
validate_shape(*
_output_shapes	
:
ą
)batch_normalization_7/gamma/Adadelta/readIdentity$batch_normalization_7/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
Ă
Hbatch_normalization_7/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_7/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_7/gamma*
dtype0*
_output_shapes
: 
˛
8batch_normalization_7/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_7/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_7/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*.
_class$
" loc:@batch_normalization_7/gamma
Ä
&batch_normalization_7/gamma/Adadelta_1
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_7/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:

-batch_normalization_7/gamma/Adadelta_1/AssignAssign&batch_normalization_7/gamma/Adadelta_18batch_normalization_7/gamma/Adadelta_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_7/gamma
ľ
+batch_normalization_7/gamma/Adadelta_1/readIdentity&batch_normalization_7/gamma/Adadelta_1*
_output_shapes	
:*
T0*.
_class$
" loc:@batch_normalization_7/gamma
ż
Ebatch_normalization_7/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_7/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
: 
¨
5batch_normalization_7/beta/Adadelta/Initializer/zerosFillEbatch_normalization_7/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_7/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
Ŕ
#batch_normalization_7/beta/Adadelta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_7/beta*
	container 

*batch_normalization_7/beta/Adadelta/AssignAssign#batch_normalization_7/beta/Adadelta5batch_normalization_7/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_7/beta*
validate_shape(*
_output_shapes	
:
Ž
(batch_normalization_7/beta/Adadelta/readIdentity#batch_normalization_7/beta/Adadelta*
_output_shapes	
:*
T0*-
_class#
!loc:@batch_normalization_7/beta
Á
Gbatch_normalization_7/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:*-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_7/beta/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes
: 
Ž
7batch_normalization_7/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_7/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_7/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
Â
%batch_normalization_7/beta/Adadelta_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *-
_class#
!loc:@batch_normalization_7/beta*
	container 

,batch_normalization_7/beta/Adadelta_1/AssignAssign%batch_normalization_7/beta/Adadelta_17batch_normalization_7/beta/Adadelta_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_7/beta
˛
*batch_normalization_7/beta/Adadelta_1/readIdentity%batch_normalization_7/beta/Adadelta_1*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:*
T0
Ż
6Variable_16/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"         @   *
_class
loc:@Variable_16*
dtype0*
_output_shapes
:

,Variable_16/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_16*
dtype0*
_output_shapes
: 
ř
&Variable_16/Adadelta/Initializer/zerosFill6Variable_16/Adadelta/Initializer/zeros/shape_as_tensor,Variable_16/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_16*'
_output_shapes
:@
ş
Variable_16/Adadelta
VariableV2*
shape:@*
dtype0*'
_output_shapes
:@*
shared_name *
_class
loc:@Variable_16*
	container 
Ţ
Variable_16/Adadelta/AssignAssignVariable_16/Adadelta&Variable_16/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*'
_output_shapes
:@

Variable_16/Adadelta/readIdentityVariable_16/Adadelta*'
_output_shapes
:@*
T0*
_class
loc:@Variable_16
ą
8Variable_16/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         @   *
_class
loc:@Variable_16*
dtype0*
_output_shapes
:

.Variable_16/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_16
ţ
(Variable_16/Adadelta_1/Initializer/zerosFill8Variable_16/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_16/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_16*'
_output_shapes
:@
ź
Variable_16/Adadelta_1
VariableV2*
dtype0*'
_output_shapes
:@*
shared_name *
_class
loc:@Variable_16*
	container *
shape:@
ä
Variable_16/Adadelta_1/AssignAssignVariable_16/Adadelta_1(Variable_16/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*'
_output_shapes
:@

Variable_16/Adadelta_1/readIdentityVariable_16/Adadelta_1*
T0*
_class
loc:@Variable_16*'
_output_shapes
:@
 
6Variable_17/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@Variable_17*
dtype0*
_output_shapes
:

,Variable_17/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
: 
ë
&Variable_17/Adadelta/Initializer/zerosFill6Variable_17/Adadelta/Initializer/zeros/shape_as_tensor,Variable_17/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_17*
_output_shapes
:@
 
Variable_17/Adadelta
VariableV2*
_output_shapes
:@*
shared_name *
_class
loc:@Variable_17*
	container *
shape:@*
dtype0
Ń
Variable_17/Adadelta/AssignAssignVariable_17/Adadelta&Variable_17/Adadelta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@Variable_17

Variable_17/Adadelta/readIdentityVariable_17/Adadelta*
T0*
_class
loc:@Variable_17*
_output_shapes
:@
˘
8Variable_17/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*
_class
loc:@Variable_17*
dtype0*
_output_shapes
:

.Variable_17/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
: 
ń
(Variable_17/Adadelta_1/Initializer/zerosFill8Variable_17/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_17/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_17*
_output_shapes
:@
˘
Variable_17/Adadelta_1
VariableV2*
_class
loc:@Variable_17*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
×
Variable_17/Adadelta_1/AssignAssignVariable_17/Adadelta_1(Variable_17/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes
:@

Variable_17/Adadelta_1/readIdentityVariable_17/Adadelta_1*
_output_shapes
:@*
T0*
_class
loc:@Variable_17
Ŕ
Fbatch_normalization_8/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_8/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
: 
Ť
6batch_normalization_8/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_8/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_8/gamma/Adadelta/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*.
_class$
" loc:@batch_normalization_8/gamma
Ŕ
$batch_normalization_8/gamma/Adadelta
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_8/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@

+batch_normalization_8/gamma/Adadelta/AssignAssign$batch_normalization_8/gamma/Adadelta6batch_normalization_8/gamma/Adadelta/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
validate_shape(*
_output_shapes
:@
°
)batch_normalization_8/gamma/Adadelta/readIdentity$batch_normalization_8/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
Â
Hbatch_normalization_8/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_8/gamma/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes
: 
ą
8batch_normalization_8/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_8/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_8/gamma/Adadelta_1/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*.
_class$
" loc:@batch_normalization_8/gamma
Â
&batch_normalization_8/gamma/Adadelta_1
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization_8/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@

-batch_normalization_8/gamma/Adadelta_1/AssignAssign&batch_normalization_8/gamma/Adadelta_18batch_normalization_8/gamma/Adadelta_1/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
´
+batch_normalization_8/gamma/Adadelta_1/readIdentity&batch_normalization_8/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes
:@
ž
Ebatch_normalization_8/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB:@*-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_8/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
_output_shapes
: 
§
5batch_normalization_8/beta/Adadelta/Initializer/zerosFillEbatch_normalization_8/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_8/beta/Adadelta/Initializer/zeros/Const*
_output_shapes
:@*
T0*

index_type0*-
_class#
!loc:@batch_normalization_8/beta
ž
#batch_normalization_8/beta/Adadelta
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_8/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@

*batch_normalization_8/beta/Adadelta/AssignAssign#batch_normalization_8/beta/Adadelta5batch_normalization_8/beta/Adadelta/Initializer/zeros*-
_class#
!loc:@batch_normalization_8/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
­
(batch_normalization_8/beta/Adadelta/readIdentity#batch_normalization_8/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
Ŕ
Gbatch_normalization_8/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
_output_shapes
:
ą
=batch_normalization_8/beta/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_8/beta*
dtype0
­
7batch_normalization_8/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_8/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_8/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@
Ŕ
%batch_normalization_8/beta/Adadelta_1
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_8/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@

,batch_normalization_8/beta/Adadelta_1/AssignAssign%batch_normalization_8/beta/Adadelta_17batch_normalization_8/beta/Adadelta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_8/beta*
validate_shape(*
_output_shapes
:@
ą
*batch_normalization_8/beta/Adadelta_1/readIdentity%batch_normalization_8/beta/Adadelta_1*
_output_shapes
:@*
T0*-
_class#
!loc:@batch_normalization_8/beta
Ż
6Variable_18/Adadelta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @       *
_class
loc:@Variable_18

,Variable_18/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_18*
dtype0
÷
&Variable_18/Adadelta/Initializer/zerosFill6Variable_18/Adadelta/Initializer/zeros/shape_as_tensor,Variable_18/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
¸
Variable_18/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_18*
	container *
shape:@ *
dtype0*&
_output_shapes
:@ 
Ý
Variable_18/Adadelta/AssignAssignVariable_18/Adadelta&Variable_18/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:@ 

Variable_18/Adadelta/readIdentityVariable_18/Adadelta*
T0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
ą
8Variable_18/Adadelta_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @       *
_class
loc:@Variable_18*
dtype0*
_output_shapes
:

.Variable_18/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_18*
dtype0*
_output_shapes
: 
ý
(Variable_18/Adadelta_1/Initializer/zerosFill8Variable_18/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_18/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:@ 
ş
Variable_18/Adadelta_1
VariableV2*
dtype0*&
_output_shapes
:@ *
shared_name *
_class
loc:@Variable_18*
	container *
shape:@ 
ă
Variable_18/Adadelta_1/AssignAssignVariable_18/Adadelta_1(Variable_18/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:@ 

Variable_18/Adadelta_1/readIdentityVariable_18/Adadelta_1*&
_output_shapes
:@ *
T0*
_class
loc:@Variable_18
 
6Variable_19/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:

,Variable_19/Adadelta/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_19*
dtype0
ë
&Variable_19/Adadelta/Initializer/zerosFill6Variable_19/Adadelta/Initializer/zeros/shape_as_tensor,Variable_19/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_19*
_output_shapes
: 
 
Variable_19/Adadelta
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@Variable_19*
	container *
shape: *
dtype0
Ń
Variable_19/Adadelta/AssignAssignVariable_19/Adadelta&Variable_19/Adadelta/Initializer/zeros*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

Variable_19/Adadelta/readIdentityVariable_19/Adadelta*
_output_shapes
: *
T0*
_class
loc:@Variable_19
˘
8Variable_19/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB: *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:

.Variable_19/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
: 
ń
(Variable_19/Adadelta_1/Initializer/zerosFill8Variable_19/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_19/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_19*
_output_shapes
: 
˘
Variable_19/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_19*
	container *
shape: *
dtype0*
_output_shapes
: 
×
Variable_19/Adadelta_1/AssignAssignVariable_19/Adadelta_1(Variable_19/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*
_output_shapes
: 

Variable_19/Adadelta_1/readIdentityVariable_19/Adadelta_1*
_output_shapes
: *
T0*
_class
loc:@Variable_19
Ŕ
Fbatch_normalization_9/gamma/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB: *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
:
ą
<batch_normalization_9/gamma/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
: 
Ť
6batch_normalization_9/gamma/Adadelta/Initializer/zerosFillFbatch_normalization_9/gamma/Adadelta/Initializer/zeros/shape_as_tensor<batch_normalization_9/gamma/Adadelta/Initializer/zeros/Const*

index_type0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: *
T0
Ŕ
$batch_normalization_9/gamma/Adadelta
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_9/gamma

+batch_normalization_9/gamma/Adadelta/AssignAssign$batch_normalization_9/gamma/Adadelta6batch_normalization_9/gamma/Adadelta/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
validate_shape(*
_output_shapes
: *
use_locking(
°
)batch_normalization_9/gamma/Adadelta/readIdentity$batch_normalization_9/gamma/Adadelta*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
Â
Hbatch_normalization_9/gamma/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB: *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0*
_output_shapes
:
ł
>batch_normalization_9/gamma/Adadelta_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@batch_normalization_9/gamma*
dtype0
ą
8batch_normalization_9/gamma/Adadelta_1/Initializer/zerosFillHbatch_normalization_9/gamma/Adadelta_1/Initializer/zeros/shape_as_tensor>batch_normalization_9/gamma/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
Â
&batch_normalization_9/gamma/Adadelta_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_9/gamma*
	container *
shape: 

-batch_normalization_9/gamma/Adadelta_1/AssignAssign&batch_normalization_9/gamma/Adadelta_18batch_normalization_9/gamma/Adadelta_1/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
validate_shape(*
_output_shapes
: *
use_locking(
´
+batch_normalization_9/gamma/Adadelta_1/readIdentity&batch_normalization_9/gamma/Adadelta_1*
T0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: 
ž
Ebatch_normalization_9/beta/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB: *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
:
Ż
;batch_normalization_9/beta/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization_9/beta*
dtype0*
_output_shapes
: 
§
5batch_normalization_9/beta/Adadelta/Initializer/zerosFillEbatch_normalization_9/beta/Adadelta/Initializer/zeros/shape_as_tensor;batch_normalization_9/beta/Adadelta/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
ž
#batch_normalization_9/beta/Adadelta
VariableV2*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_9/beta*
	container *
shape: *
dtype0

*batch_normalization_9/beta/Adadelta/AssignAssign#batch_normalization_9/beta/Adadelta5batch_normalization_9/beta/Adadelta/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_9/beta*
validate_shape(*
_output_shapes
: 
­
(batch_normalization_9/beta/Adadelta/readIdentity#batch_normalization_9/beta/Adadelta*
T0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
Ŕ
Gbatch_normalization_9/beta/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB: *-
_class#
!loc:@batch_normalization_9/beta
ą
=batch_normalization_9/beta/Adadelta_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *-
_class#
!loc:@batch_normalization_9/beta
­
7batch_normalization_9/beta/Adadelta_1/Initializer/zerosFillGbatch_normalization_9/beta/Adadelta_1/Initializer/zeros/shape_as_tensor=batch_normalization_9/beta/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
Ŕ
%batch_normalization_9/beta/Adadelta_1
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization_9/beta*
	container *
shape: *
dtype0*
_output_shapes
: 

,batch_normalization_9/beta/Adadelta_1/AssignAssign%batch_normalization_9/beta/Adadelta_17batch_normalization_9/beta/Adadelta_1/Initializer/zeros*-
_class#
!loc:@batch_normalization_9/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
ą
*batch_normalization_9/beta/Adadelta_1/readIdentity%batch_normalization_9/beta/Adadelta_1*
T0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 

6Variable_20/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 

,Variable_20/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 
ç
&Variable_20/Adadelta/Initializer/zerosFill6Variable_20/Adadelta/Initializer/zeros/shape_as_tensor,Variable_20/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_20*
_output_shapes
: 

Variable_20/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_20*
	container *
shape: *
dtype0*
_output_shapes
: 
Í
Variable_20/Adadelta/AssignAssignVariable_20/Adadelta&Variable_20/Adadelta/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(
|
Variable_20/Adadelta/readIdentityVariable_20/Adadelta*
T0*
_class
loc:@Variable_20*
_output_shapes
: 

8Variable_20/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 

.Variable_20/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 
í
(Variable_20/Adadelta_1/Initializer/zerosFill8Variable_20/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_20/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_20*
_output_shapes
: 

Variable_20/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_20*
	container *
shape: *
dtype0*
_output_shapes
: 
Ó
Variable_20/Adadelta_1/AssignAssignVariable_20/Adadelta_1(Variable_20/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*
_output_shapes
: 

Variable_20/Adadelta_1/readIdentityVariable_20/Adadelta_1*
_class
loc:@Variable_20*
_output_shapes
: *
T0

6Variable_21/Adadelta/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 

,Variable_21/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 
ç
&Variable_21/Adadelta/Initializer/zerosFill6Variable_21/Adadelta/Initializer/zeros/shape_as_tensor,Variable_21/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_21*
_output_shapes
: 

Variable_21/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_21*
	container *
shape: *
dtype0*
_output_shapes
: 
Í
Variable_21/Adadelta/AssignAssignVariable_21/Adadelta&Variable_21/Adadelta/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
: 
|
Variable_21/Adadelta/readIdentityVariable_21/Adadelta*
_output_shapes
: *
T0*
_class
loc:@Variable_21

8Variable_21/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 

.Variable_21/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 
í
(Variable_21/Adadelta_1/Initializer/zerosFill8Variable_21/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_21/Adadelta_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_21*
_output_shapes
: 

Variable_21/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_21*
	container *
shape: *
dtype0*
_output_shapes
: 
Ó
Variable_21/Adadelta_1/AssignAssignVariable_21/Adadelta_1(Variable_21/Adadelta_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*
_output_shapes
: 

Variable_21/Adadelta_1/readIdentityVariable_21/Adadelta_1*
T0*
_class
loc:@Variable_21*
_output_shapes
: 
Ż
6Variable_22/Adadelta/Initializer/zeros/shape_as_tensorConst*%
valueB"             *
_class
loc:@Variable_22*
dtype0*
_output_shapes
:

,Variable_22/Adadelta/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_22*
dtype0*
_output_shapes
: 
÷
&Variable_22/Adadelta/Initializer/zerosFill6Variable_22/Adadelta/Initializer/zeros/shape_as_tensor,Variable_22/Adadelta/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_22*&
_output_shapes
: 
¸
Variable_22/Adadelta
VariableV2*
shared_name *
_class
loc:@Variable_22*
	container *
shape: *
dtype0*&
_output_shapes
: 
Ý
Variable_22/Adadelta/AssignAssignVariable_22/Adadelta&Variable_22/Adadelta/Initializer/zeros*&
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(

Variable_22/Adadelta/readIdentityVariable_22/Adadelta*
T0*
_class
loc:@Variable_22*&
_output_shapes
: 
ą
8Variable_22/Adadelta_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"             *
_class
loc:@Variable_22*
dtype0

.Variable_22/Adadelta_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_22*
dtype0*
_output_shapes
: 
ý
(Variable_22/Adadelta_1/Initializer/zerosFill8Variable_22/Adadelta_1/Initializer/zeros/shape_as_tensor.Variable_22/Adadelta_1/Initializer/zeros/Const*&
_output_shapes
: *
T0*

index_type0*
_class
loc:@Variable_22
ş
Variable_22/Adadelta_1
VariableV2*
shared_name *
_class
loc:@Variable_22*
	container *
shape: *
dtype0*&
_output_shapes
: 
ă
Variable_22/Adadelta_1/AssignAssignVariable_22/Adadelta_1(Variable_22/Adadelta_1/Initializer/zeros*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
: *
use_locking(

Variable_22/Adadelta_1/readIdentityVariable_22/Adadelta_1*
T0*
_class
loc:@Variable_22*&
_output_shapes
: 
P
Adadelta/lrConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adadelta/rhoConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
U
Adadelta/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
Š
&Adadelta/update_Variable/ApplyAdadeltaApplyAdadeltaVariableVariable/AdadeltaVariable/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable*&
_output_shapes
:@
¨
(Adadelta/update_Variable_1/ApplyAdadeltaApplyAdadelta
Variable_1Variable_1/AdadeltaVariable_1/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@Variable_1
Ó
7Adadelta/update_batch_normalization/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization/gamma"batch_normalization/gamma/Adadelta$batch_normalization/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_29*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
Î
6Adadelta/update_batch_normalization/beta/ApplyAdadeltaApplyAdadeltabatch_normalization/beta!batch_normalization/beta/Adadelta#batch_normalization/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_30*
use_locking( *
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
ś
(Adadelta/update_Variable_2/ApplyAdadeltaApplyAdadelta
Variable_2Variable_2/AdadeltaVariable_2/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*'
_output_shapes
:@
Ť
(Adadelta/update_Variable_3/ApplyAdadeltaApplyAdadelta
Variable_3Variable_3/AdadeltaVariable_3/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@Variable_3
Ţ
9Adadelta/update_batch_normalization_1/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_1/gamma$batch_normalization_1/gamma/Adadelta&batch_normalization_1/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_26*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes	
:*
use_locking( *
T0
Ů
8Adadelta/update_batch_normalization_1/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_1/beta#batch_normalization_1/beta/Adadelta%batch_normalization_1/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_27*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes	
:
ˇ
(Adadelta/update_Variable_4/ApplyAdadeltaApplyAdadelta
Variable_4Variable_4/AdadeltaVariable_4/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_4*(
_output_shapes
:
Ť
(Adadelta/update_Variable_5/ApplyAdadeltaApplyAdadelta
Variable_5Variable_5/AdadeltaVariable_5/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@Variable_5
Ţ
9Adadelta/update_batch_normalization_2/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_2/gamma$batch_normalization_2/gamma/Adadelta&batch_normalization_2/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_23*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes	
:
Ů
8Adadelta/update_batch_normalization_2/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_2/beta#batch_normalization_2/beta/Adadelta%batch_normalization_2/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_24*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes	
:
ˇ
(Adadelta/update_Variable_6/ApplyAdadeltaApplyAdadelta
Variable_6Variable_6/AdadeltaVariable_6/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*(
_output_shapes
:
Ť
(Adadelta/update_Variable_7/ApplyAdadeltaApplyAdadelta
Variable_7Variable_7/AdadeltaVariable_7/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@Variable_7
Ţ
9Adadelta/update_batch_normalization_3/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_3/gamma$batch_normalization_3/gamma/Adadelta&batch_normalization_3/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_20*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:*
use_locking( 
Ů
8Adadelta/update_batch_normalization_3/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_3/beta#batch_normalization_3/beta/Adadelta%batch_normalization_3/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_21*
_output_shapes	
:*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta
ˇ
(Adadelta/update_Variable_8/ApplyAdadeltaApplyAdadelta
Variable_8Variable_8/AdadeltaVariable_8/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_8*(
_output_shapes
:*
use_locking( 
Ť
(Adadelta/update_Variable_9/ApplyAdadeltaApplyAdadelta
Variable_9Variable_9/AdadeltaVariable_9/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_9*
_output_shapes	
:
Ţ
9Adadelta/update_batch_normalization_4/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_4/gamma$batch_normalization_4/gamma/Adadelta&batch_normalization_4/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_17*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes	
:
Ů
8Adadelta/update_batch_normalization_4/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_4/beta#batch_normalization_4/beta/Adadelta%batch_normalization_4/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_18*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes	
:
ź
)Adadelta/update_Variable_10/ApplyAdadeltaApplyAdadeltaVariable_10Variable_10/AdadeltaVariable_10/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_5_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_10*(
_output_shapes
:
°
)Adadelta/update_Variable_11/ApplyAdadeltaApplyAdadeltaVariable_11Variable_11/AdadeltaVariable_11/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_5_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*
_class
loc:@Variable_11
Ţ
9Adadelta/update_batch_normalization_5/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_5/gamma$batch_normalization_5/gamma/Adadelta&batch_normalization_5/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_14*
_output_shapes	
:*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_5/gamma
Ů
8Adadelta/update_batch_normalization_5/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_5/beta#batch_normalization_5/beta/Adadelta%batch_normalization_5/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_15*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes	
:
ź
)Adadelta/update_Variable_12/ApplyAdadeltaApplyAdadeltaVariable_12Variable_12/AdadeltaVariable_12/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_6_grad/tuple/control_dependency_1*(
_output_shapes
:*
use_locking( *
T0*
_class
loc:@Variable_12
°
)Adadelta/update_Variable_13/ApplyAdadeltaApplyAdadeltaVariable_13Variable_13/AdadeltaVariable_13/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_6_grad/tuple/control_dependency_1*
_class
loc:@Variable_13*
_output_shapes	
:*
use_locking( *
T0
Ţ
9Adadelta/update_batch_normalization_6/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_6/gamma$batch_normalization_6/gamma/Adadelta&batch_normalization_6/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_11*
_output_shapes	
:*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_6/gamma
Ů
8Adadelta/update_batch_normalization_6/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_6/beta#batch_normalization_6/beta/Adadelta%batch_normalization_6/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_12*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes	
:
ź
)Adadelta/update_Variable_14/ApplyAdadeltaApplyAdadeltaVariable_14Variable_14/AdadeltaVariable_14/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_7_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_14*(
_output_shapes
:*
use_locking( 
°
)Adadelta/update_Variable_15/ApplyAdadeltaApplyAdadeltaVariable_15Variable_15/AdadeltaVariable_15/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_7_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_15*
_output_shapes	
:
Ý
9Adadelta/update_batch_normalization_7/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_7/gamma$batch_normalization_7/gamma/Adadelta&batch_normalization_7/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_8*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes	
:
Ř
8Adadelta/update_batch_normalization_7/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_7/beta#batch_normalization_7/beta/Adadelta%batch_normalization_7/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_9*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes	
:
ť
)Adadelta/update_Variable_16/ApplyAdadeltaApplyAdadeltaVariable_16Variable_16/AdadeltaVariable_16/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_8_grad/tuple/control_dependency_1*'
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@Variable_16
Ż
)Adadelta/update_Variable_17/ApplyAdadeltaApplyAdadeltaVariable_17Variable_17/AdadeltaVariable_17/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_17*
_output_shapes
:@
Ü
9Adadelta/update_batch_normalization_8/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_8/gamma$batch_normalization_8/gamma/Adadelta&batch_normalization_8/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_5*
_output_shapes
:@*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_8/gamma
×
8Adadelta/update_batch_normalization_8/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_8/beta#batch_normalization_8/beta/Adadelta%batch_normalization_8/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_6*-
_class#
!loc:@batch_normalization_8/beta*
_output_shapes
:@*
use_locking( *
T0
ş
)Adadelta/update_Variable_18/ApplyAdadeltaApplyAdadeltaVariable_18Variable_18/AdadeltaVariable_18/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon2gradients/Conv2D_9_grad/tuple/control_dependency_1*&
_output_shapes
:@ *
use_locking( *
T0*
_class
loc:@Variable_18
Ż
)Adadelta/update_Variable_19/ApplyAdadeltaApplyAdadeltaVariable_19Variable_19/AdadeltaVariable_19/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/BiasAdd_9_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_19*
_output_shapes
: 
Ü
9Adadelta/update_batch_normalization_9/gamma/ApplyAdadeltaApplyAdadeltabatch_normalization_9/gamma$batch_normalization_9/gamma/Adadelta&batch_normalization_9/gamma/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_2*
_output_shapes
: *
use_locking( *
T0*.
_class$
" loc:@batch_normalization_9/gamma
×
8Adadelta/update_batch_normalization_9/beta/ApplyAdadeltaApplyAdadeltabatch_normalization_9/beta#batch_normalization_9/beta/Adadelta%batch_normalization_9/beta/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilongradients/AddN_3*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
Ľ
)Adadelta/update_Variable_20/ApplyAdadeltaApplyAdadeltaVariable_20Variable_20/AdadeltaVariable_20/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon-gradients/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_20*
_output_shapes
: 
Ľ
)Adadelta/update_Variable_21/ApplyAdadeltaApplyAdadeltaVariable_21Variable_21/AdadeltaVariable_21/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_21*
_output_shapes
: 
ť
)Adadelta/update_Variable_22/ApplyAdadeltaApplyAdadeltaVariable_22Variable_22/AdadeltaVariable_22/Adadelta_1Adadelta/lrAdadelta/rhoAdadelta/epsilon3gradients/Conv2D_10_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_22*&
_output_shapes
: 

AdadeltaNoOp'^Adadelta/update_Variable/ApplyAdadelta)^Adadelta/update_Variable_1/ApplyAdadelta8^Adadelta/update_batch_normalization/gamma/ApplyAdadelta7^Adadelta/update_batch_normalization/beta/ApplyAdadelta)^Adadelta/update_Variable_2/ApplyAdadelta)^Adadelta/update_Variable_3/ApplyAdadelta:^Adadelta/update_batch_normalization_1/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_1/beta/ApplyAdadelta)^Adadelta/update_Variable_4/ApplyAdadelta)^Adadelta/update_Variable_5/ApplyAdadelta:^Adadelta/update_batch_normalization_2/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_2/beta/ApplyAdadelta)^Adadelta/update_Variable_6/ApplyAdadelta)^Adadelta/update_Variable_7/ApplyAdadelta:^Adadelta/update_batch_normalization_3/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_3/beta/ApplyAdadelta)^Adadelta/update_Variable_8/ApplyAdadelta)^Adadelta/update_Variable_9/ApplyAdadelta:^Adadelta/update_batch_normalization_4/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_4/beta/ApplyAdadelta*^Adadelta/update_Variable_10/ApplyAdadelta*^Adadelta/update_Variable_11/ApplyAdadelta:^Adadelta/update_batch_normalization_5/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_5/beta/ApplyAdadelta*^Adadelta/update_Variable_12/ApplyAdadelta*^Adadelta/update_Variable_13/ApplyAdadelta:^Adadelta/update_batch_normalization_6/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_6/beta/ApplyAdadelta*^Adadelta/update_Variable_14/ApplyAdadelta*^Adadelta/update_Variable_15/ApplyAdadelta:^Adadelta/update_batch_normalization_7/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_7/beta/ApplyAdadelta*^Adadelta/update_Variable_16/ApplyAdadelta*^Adadelta/update_Variable_17/ApplyAdadelta:^Adadelta/update_batch_normalization_8/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_8/beta/ApplyAdadelta*^Adadelta/update_Variable_18/ApplyAdadelta*^Adadelta/update_Variable_19/ApplyAdadelta:^Adadelta/update_batch_normalization_9/gamma/ApplyAdadelta9^Adadelta/update_batch_normalization_9/beta/ApplyAdadelta*^Adadelta/update_Variable_20/ApplyAdadelta*^Adadelta/update_Variable_21/ApplyAdadelta*^Adadelta/update_Variable_22/ApplyAdadelta
I
Merge/MergeSummaryMergeSummaryloss*
N*
_output_shapes
: ""
train_op


Adadelta"Ü
cond_contextÜÜ
Ź
"batch_normalization/cond/cond_text"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_t:0 *ş
Conv2D:0
batch_normalization/beta/read:0
 batch_normalization/cond/Const:0
"batch_normalization/cond/Const_1:0
0batch_normalization/cond/FusedBatchNorm/Switch:1
2batch_normalization/cond/FusedBatchNorm/Switch_1:1
2batch_normalization/cond/FusedBatchNorm/Switch_2:1
)batch_normalization/cond/FusedBatchNorm:0
)batch_normalization/cond/FusedBatchNorm:1
)batch_normalization/cond/FusedBatchNorm:2
)batch_normalization/cond/FusedBatchNorm:3
)batch_normalization/cond/FusedBatchNorm:4
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_t:0
 batch_normalization/gamma/read:0V
 batch_normalization/gamma/read:02batch_normalization/cond/FusedBatchNorm/Switch_1:1U
batch_normalization/beta/read:02batch_normalization/cond/FusedBatchNorm/Switch_2:1<
Conv2D:00batch_normalization/cond/FusedBatchNorm/Switch:1


$batch_normalization/cond/cond_text_1"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_f:0*	
Conv2D:0
batch_normalization/beta/read:0
2batch_normalization/cond/FusedBatchNorm_1/Switch:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_1:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_2:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_3:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_4:0
+batch_normalization/cond/FusedBatchNorm_1:0
+batch_normalization/cond/FusedBatchNorm_1:1
+batch_normalization/cond/FusedBatchNorm_1:2
+batch_normalization/cond/FusedBatchNorm_1:3
+batch_normalization/cond/FusedBatchNorm_1:4
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_f:0
 batch_normalization/gamma/read:0
&batch_normalization/moving_mean/read:0
*batch_normalization/moving_variance/read:0b
*batch_normalization/moving_variance/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_4:0^
&batch_normalization/moving_mean/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_3:0X
 batch_normalization/gamma/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_1:0W
batch_normalization/beta/read:04batch_normalization/cond/FusedBatchNorm_1/Switch_2:0>
Conv2D:02batch_normalization/cond/FusedBatchNorm_1/Switch:0
č
$batch_normalization/cond_1/cond_text$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_t:0 *q
"batch_normalization/cond_1/Const:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_t:0
ę
&batch_normalization/cond_1/cond_text_1$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_f:0*s
$batch_normalization/cond_1/Const_1:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_f:0
Ü
$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *ä

Conv2D_1:0
!batch_normalization_1/beta/read:0
"batch_normalization_1/cond/Const:0
$batch_normalization_1/cond/Const_1:0
2batch_normalization_1/cond/FusedBatchNorm/Switch:1
4batch_normalization_1/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_1/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_1/cond/FusedBatchNorm:0
+batch_normalization_1/cond/FusedBatchNorm:1
+batch_normalization_1/cond/FusedBatchNorm:2
+batch_normalization_1/cond/FusedBatchNorm:3
+batch_normalization_1/cond/FusedBatchNorm:4
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0
"batch_normalization_1/gamma/read:0@

Conv2D_1:02batch_normalization_1/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_1/gamma/read:04batch_normalization_1/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_1/beta/read:04batch_normalization_1/cond/FusedBatchNorm/Switch_2:1
ź

&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*Ä	

Conv2D_1:0
!batch_normalization_1/beta/read:0
4batch_normalization_1/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_1/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_1/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_1/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_1/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_1/cond/FusedBatchNorm_1:0
-batch_normalization_1/cond/FusedBatchNorm_1:1
-batch_normalization_1/cond/FusedBatchNorm_1:2
-batch_normalization_1/cond/FusedBatchNorm_1:3
-batch_normalization_1/cond/FusedBatchNorm_1:4
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
"batch_normalization_1/gamma/read:0
(batch_normalization_1/moving_mean/read:0
,batch_normalization_1/moving_variance/read:0[
!batch_normalization_1/beta/read:06batch_normalization_1/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_1:04batch_normalization_1/cond/FusedBatchNorm_1/Switch:0f
,batch_normalization_1/moving_variance/read:06batch_normalization_1/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_1/moving_mean/read:06batch_normalization_1/cond/FusedBatchNorm_1/Switch_3:0\
"batch_normalization_1/gamma/read:06batch_normalization_1/cond/FusedBatchNorm_1/Switch_1:0
ô
&batch_normalization_1/cond_1/cond_text&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_t:0 *w
$batch_normalization_1/cond_1/Const:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_t:0
ö
(batch_normalization_1/cond_1/cond_text_1&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_f:0*y
&batch_normalization_1/cond_1/Const_1:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_f:0
Ü
$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *ä

Conv2D_2:0
!batch_normalization_2/beta/read:0
"batch_normalization_2/cond/Const:0
$batch_normalization_2/cond/Const_1:0
2batch_normalization_2/cond/FusedBatchNorm/Switch:1
4batch_normalization_2/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_2/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_2/cond/FusedBatchNorm:0
+batch_normalization_2/cond/FusedBatchNorm:1
+batch_normalization_2/cond/FusedBatchNorm:2
+batch_normalization_2/cond/FusedBatchNorm:3
+batch_normalization_2/cond/FusedBatchNorm:4
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0
"batch_normalization_2/gamma/read:0@

Conv2D_2:02batch_normalization_2/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_2/gamma/read:04batch_normalization_2/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_2/beta/read:04batch_normalization_2/cond/FusedBatchNorm/Switch_2:1
ź

&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*Ä	

Conv2D_2:0
!batch_normalization_2/beta/read:0
4batch_normalization_2/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_2/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_2/cond/FusedBatchNorm_1:0
-batch_normalization_2/cond/FusedBatchNorm_1:1
-batch_normalization_2/cond/FusedBatchNorm_1:2
-batch_normalization_2/cond/FusedBatchNorm_1:3
-batch_normalization_2/cond/FusedBatchNorm_1:4
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
"batch_normalization_2/gamma/read:0
(batch_normalization_2/moving_mean/read:0
,batch_normalization_2/moving_variance/read:0\
"batch_normalization_2/gamma/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_1:0f
,batch_normalization_2/moving_variance/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_2/moving_mean/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_3:0[
!batch_normalization_2/beta/read:06batch_normalization_2/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_2:04batch_normalization_2/cond/FusedBatchNorm_1/Switch:0
ô
&batch_normalization_2/cond_1/cond_text&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_t:0 *w
$batch_normalization_2/cond_1/Const:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_t:0
ö
(batch_normalization_2/cond_1/cond_text_1&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_f:0*y
&batch_normalization_2/cond_1/Const_1:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_f:0
Ü
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *ä

Conv2D_3:0
!batch_normalization_3/beta/read:0
"batch_normalization_3/cond/Const:0
$batch_normalization_3/cond/Const_1:0
2batch_normalization_3/cond/FusedBatchNorm/Switch:1
4batch_normalization_3/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_3/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_3/cond/FusedBatchNorm:0
+batch_normalization_3/cond/FusedBatchNorm:1
+batch_normalization_3/cond/FusedBatchNorm:2
+batch_normalization_3/cond/FusedBatchNorm:3
+batch_normalization_3/cond/FusedBatchNorm:4
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0
"batch_normalization_3/gamma/read:0@

Conv2D_3:02batch_normalization_3/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_3/gamma/read:04batch_normalization_3/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_3/beta/read:04batch_normalization_3/cond/FusedBatchNorm/Switch_2:1
ź

&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*Ä	

Conv2D_3:0
!batch_normalization_3/beta/read:0
4batch_normalization_3/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_3/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_3/cond/FusedBatchNorm_1:0
-batch_normalization_3/cond/FusedBatchNorm_1:1
-batch_normalization_3/cond/FusedBatchNorm_1:2
-batch_normalization_3/cond/FusedBatchNorm_1:3
-batch_normalization_3/cond/FusedBatchNorm_1:4
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
"batch_normalization_3/gamma/read:0
(batch_normalization_3/moving_mean/read:0
,batch_normalization_3/moving_variance/read:0\
"batch_normalization_3/gamma/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_1:0[
!batch_normalization_3/beta/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_3:04batch_normalization_3/cond/FusedBatchNorm_1/Switch:0f
,batch_normalization_3/moving_variance/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_3/moving_mean/read:06batch_normalization_3/cond/FusedBatchNorm_1/Switch_3:0
ô
&batch_normalization_3/cond_1/cond_text&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_t:0 *w
$batch_normalization_3/cond_1/Const:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_t:0
ö
(batch_normalization_3/cond_1/cond_text_1&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_f:0*y
&batch_normalization_3/cond_1/Const_1:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_f:0
Ü
$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *ä

Conv2D_4:0
!batch_normalization_4/beta/read:0
"batch_normalization_4/cond/Const:0
$batch_normalization_4/cond/Const_1:0
2batch_normalization_4/cond/FusedBatchNorm/Switch:1
4batch_normalization_4/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_4/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_4/cond/FusedBatchNorm:0
+batch_normalization_4/cond/FusedBatchNorm:1
+batch_normalization_4/cond/FusedBatchNorm:2
+batch_normalization_4/cond/FusedBatchNorm:3
+batch_normalization_4/cond/FusedBatchNorm:4
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0
"batch_normalization_4/gamma/read:0Y
!batch_normalization_4/beta/read:04batch_normalization_4/cond/FusedBatchNorm/Switch_2:1@

Conv2D_4:02batch_normalization_4/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_4/gamma/read:04batch_normalization_4/cond/FusedBatchNorm/Switch_1:1
ź

&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*Ä	

Conv2D_4:0
!batch_normalization_4/beta/read:0
4batch_normalization_4/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_4/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_4/cond/FusedBatchNorm_1:0
-batch_normalization_4/cond/FusedBatchNorm_1:1
-batch_normalization_4/cond/FusedBatchNorm_1:2
-batch_normalization_4/cond/FusedBatchNorm_1:3
-batch_normalization_4/cond/FusedBatchNorm_1:4
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
"batch_normalization_4/gamma/read:0
(batch_normalization_4/moving_mean/read:0
,batch_normalization_4/moving_variance/read:0f
,batch_normalization_4/moving_variance/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_4/moving_mean/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_3:0[
!batch_normalization_4/beta/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_4:04batch_normalization_4/cond/FusedBatchNorm_1/Switch:0\
"batch_normalization_4/gamma/read:06batch_normalization_4/cond/FusedBatchNorm_1/Switch_1:0
ô
&batch_normalization_4/cond_1/cond_text&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_t:0 *w
$batch_normalization_4/cond_1/Const:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_t:0
ö
(batch_normalization_4/cond_1/cond_text_1&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_f:0*y
&batch_normalization_4/cond_1/Const_1:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_f:0
Ü
$batch_normalization_5/cond/cond_text$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_t:0 *ä

Conv2D_5:0
!batch_normalization_5/beta/read:0
"batch_normalization_5/cond/Const:0
$batch_normalization_5/cond/Const_1:0
2batch_normalization_5/cond/FusedBatchNorm/Switch:1
4batch_normalization_5/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_5/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_5/cond/FusedBatchNorm:0
+batch_normalization_5/cond/FusedBatchNorm:1
+batch_normalization_5/cond/FusedBatchNorm:2
+batch_normalization_5/cond/FusedBatchNorm:3
+batch_normalization_5/cond/FusedBatchNorm:4
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_t:0
"batch_normalization_5/gamma/read:0Y
!batch_normalization_5/beta/read:04batch_normalization_5/cond/FusedBatchNorm/Switch_2:1@

Conv2D_5:02batch_normalization_5/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_5/gamma/read:04batch_normalization_5/cond/FusedBatchNorm/Switch_1:1
ź

&batch_normalization_5/cond/cond_text_1$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_f:0*Ä	

Conv2D_5:0
!batch_normalization_5/beta/read:0
4batch_normalization_5/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_5/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_5/cond/FusedBatchNorm_1:0
-batch_normalization_5/cond/FusedBatchNorm_1:1
-batch_normalization_5/cond/FusedBatchNorm_1:2
-batch_normalization_5/cond/FusedBatchNorm_1:3
-batch_normalization_5/cond/FusedBatchNorm_1:4
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_f:0
"batch_normalization_5/gamma/read:0
(batch_normalization_5/moving_mean/read:0
,batch_normalization_5/moving_variance/read:0\
"batch_normalization_5/gamma/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_1:0[
!batch_normalization_5/beta/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_5:04batch_normalization_5/cond/FusedBatchNorm_1/Switch:0f
,batch_normalization_5/moving_variance/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_5/moving_mean/read:06batch_normalization_5/cond/FusedBatchNorm_1/Switch_3:0
ô
&batch_normalization_5/cond_1/cond_text&batch_normalization_5/cond_1/pred_id:0'batch_normalization_5/cond_1/switch_t:0 *w
$batch_normalization_5/cond_1/Const:0
&batch_normalization_5/cond_1/pred_id:0
'batch_normalization_5/cond_1/switch_t:0
ö
(batch_normalization_5/cond_1/cond_text_1&batch_normalization_5/cond_1/pred_id:0'batch_normalization_5/cond_1/switch_f:0*y
&batch_normalization_5/cond_1/Const_1:0
&batch_normalization_5/cond_1/pred_id:0
'batch_normalization_5/cond_1/switch_f:0
Ü
$batch_normalization_6/cond/cond_text$batch_normalization_6/cond/pred_id:0%batch_normalization_6/cond/switch_t:0 *ä

Conv2D_6:0
!batch_normalization_6/beta/read:0
"batch_normalization_6/cond/Const:0
$batch_normalization_6/cond/Const_1:0
2batch_normalization_6/cond/FusedBatchNorm/Switch:1
4batch_normalization_6/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_6/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_6/cond/FusedBatchNorm:0
+batch_normalization_6/cond/FusedBatchNorm:1
+batch_normalization_6/cond/FusedBatchNorm:2
+batch_normalization_6/cond/FusedBatchNorm:3
+batch_normalization_6/cond/FusedBatchNorm:4
$batch_normalization_6/cond/pred_id:0
%batch_normalization_6/cond/switch_t:0
"batch_normalization_6/gamma/read:0@

Conv2D_6:02batch_normalization_6/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_6/gamma/read:04batch_normalization_6/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_6/beta/read:04batch_normalization_6/cond/FusedBatchNorm/Switch_2:1
ź

&batch_normalization_6/cond/cond_text_1$batch_normalization_6/cond/pred_id:0%batch_normalization_6/cond/switch_f:0*Ä	

Conv2D_6:0
!batch_normalization_6/beta/read:0
4batch_normalization_6/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_6/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_6/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_6/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_6/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_6/cond/FusedBatchNorm_1:0
-batch_normalization_6/cond/FusedBatchNorm_1:1
-batch_normalization_6/cond/FusedBatchNorm_1:2
-batch_normalization_6/cond/FusedBatchNorm_1:3
-batch_normalization_6/cond/FusedBatchNorm_1:4
$batch_normalization_6/cond/pred_id:0
%batch_normalization_6/cond/switch_f:0
"batch_normalization_6/gamma/read:0
(batch_normalization_6/moving_mean/read:0
,batch_normalization_6/moving_variance/read:0f
,batch_normalization_6/moving_variance/read:06batch_normalization_6/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_6/moving_mean/read:06batch_normalization_6/cond/FusedBatchNorm_1/Switch_3:0[
!batch_normalization_6/beta/read:06batch_normalization_6/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_6:04batch_normalization_6/cond/FusedBatchNorm_1/Switch:0\
"batch_normalization_6/gamma/read:06batch_normalization_6/cond/FusedBatchNorm_1/Switch_1:0
ô
&batch_normalization_6/cond_1/cond_text&batch_normalization_6/cond_1/pred_id:0'batch_normalization_6/cond_1/switch_t:0 *w
$batch_normalization_6/cond_1/Const:0
&batch_normalization_6/cond_1/pred_id:0
'batch_normalization_6/cond_1/switch_t:0
ö
(batch_normalization_6/cond_1/cond_text_1&batch_normalization_6/cond_1/pred_id:0'batch_normalization_6/cond_1/switch_f:0*y
&batch_normalization_6/cond_1/Const_1:0
&batch_normalization_6/cond_1/pred_id:0
'batch_normalization_6/cond_1/switch_f:0
Ü
$batch_normalization_7/cond/cond_text$batch_normalization_7/cond/pred_id:0%batch_normalization_7/cond/switch_t:0 *ä

Conv2D_7:0
!batch_normalization_7/beta/read:0
"batch_normalization_7/cond/Const:0
$batch_normalization_7/cond/Const_1:0
2batch_normalization_7/cond/FusedBatchNorm/Switch:1
4batch_normalization_7/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_7/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_7/cond/FusedBatchNorm:0
+batch_normalization_7/cond/FusedBatchNorm:1
+batch_normalization_7/cond/FusedBatchNorm:2
+batch_normalization_7/cond/FusedBatchNorm:3
+batch_normalization_7/cond/FusedBatchNorm:4
$batch_normalization_7/cond/pred_id:0
%batch_normalization_7/cond/switch_t:0
"batch_normalization_7/gamma/read:0@

Conv2D_7:02batch_normalization_7/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_7/gamma/read:04batch_normalization_7/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_7/beta/read:04batch_normalization_7/cond/FusedBatchNorm/Switch_2:1
ź

&batch_normalization_7/cond/cond_text_1$batch_normalization_7/cond/pred_id:0%batch_normalization_7/cond/switch_f:0*Ä	

Conv2D_7:0
!batch_normalization_7/beta/read:0
4batch_normalization_7/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_7/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_7/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_7/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_7/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_7/cond/FusedBatchNorm_1:0
-batch_normalization_7/cond/FusedBatchNorm_1:1
-batch_normalization_7/cond/FusedBatchNorm_1:2
-batch_normalization_7/cond/FusedBatchNorm_1:3
-batch_normalization_7/cond/FusedBatchNorm_1:4
$batch_normalization_7/cond/pred_id:0
%batch_normalization_7/cond/switch_f:0
"batch_normalization_7/gamma/read:0
(batch_normalization_7/moving_mean/read:0
,batch_normalization_7/moving_variance/read:0\
"batch_normalization_7/gamma/read:06batch_normalization_7/cond/FusedBatchNorm_1/Switch_1:0[
!batch_normalization_7/beta/read:06batch_normalization_7/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_7:04batch_normalization_7/cond/FusedBatchNorm_1/Switch:0f
,batch_normalization_7/moving_variance/read:06batch_normalization_7/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_7/moving_mean/read:06batch_normalization_7/cond/FusedBatchNorm_1/Switch_3:0
ô
&batch_normalization_7/cond_1/cond_text&batch_normalization_7/cond_1/pred_id:0'batch_normalization_7/cond_1/switch_t:0 *w
$batch_normalization_7/cond_1/Const:0
&batch_normalization_7/cond_1/pred_id:0
'batch_normalization_7/cond_1/switch_t:0
ö
(batch_normalization_7/cond_1/cond_text_1&batch_normalization_7/cond_1/pred_id:0'batch_normalization_7/cond_1/switch_f:0*y
&batch_normalization_7/cond_1/Const_1:0
&batch_normalization_7/cond_1/pred_id:0
'batch_normalization_7/cond_1/switch_f:0
Ü
$batch_normalization_8/cond/cond_text$batch_normalization_8/cond/pred_id:0%batch_normalization_8/cond/switch_t:0 *ä

Conv2D_8:0
!batch_normalization_8/beta/read:0
"batch_normalization_8/cond/Const:0
$batch_normalization_8/cond/Const_1:0
2batch_normalization_8/cond/FusedBatchNorm/Switch:1
4batch_normalization_8/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_8/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_8/cond/FusedBatchNorm:0
+batch_normalization_8/cond/FusedBatchNorm:1
+batch_normalization_8/cond/FusedBatchNorm:2
+batch_normalization_8/cond/FusedBatchNorm:3
+batch_normalization_8/cond/FusedBatchNorm:4
$batch_normalization_8/cond/pred_id:0
%batch_normalization_8/cond/switch_t:0
"batch_normalization_8/gamma/read:0Y
!batch_normalization_8/beta/read:04batch_normalization_8/cond/FusedBatchNorm/Switch_2:1@

Conv2D_8:02batch_normalization_8/cond/FusedBatchNorm/Switch:1Z
"batch_normalization_8/gamma/read:04batch_normalization_8/cond/FusedBatchNorm/Switch_1:1
ź

&batch_normalization_8/cond/cond_text_1$batch_normalization_8/cond/pred_id:0%batch_normalization_8/cond/switch_f:0*Ä	

Conv2D_8:0
!batch_normalization_8/beta/read:0
4batch_normalization_8/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_8/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_8/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_8/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_8/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_8/cond/FusedBatchNorm_1:0
-batch_normalization_8/cond/FusedBatchNorm_1:1
-batch_normalization_8/cond/FusedBatchNorm_1:2
-batch_normalization_8/cond/FusedBatchNorm_1:3
-batch_normalization_8/cond/FusedBatchNorm_1:4
$batch_normalization_8/cond/pred_id:0
%batch_normalization_8/cond/switch_f:0
"batch_normalization_8/gamma/read:0
(batch_normalization_8/moving_mean/read:0
,batch_normalization_8/moving_variance/read:0f
,batch_normalization_8/moving_variance/read:06batch_normalization_8/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_8/moving_mean/read:06batch_normalization_8/cond/FusedBatchNorm_1/Switch_3:0[
!batch_normalization_8/beta/read:06batch_normalization_8/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_8:04batch_normalization_8/cond/FusedBatchNorm_1/Switch:0\
"batch_normalization_8/gamma/read:06batch_normalization_8/cond/FusedBatchNorm_1/Switch_1:0
ô
&batch_normalization_8/cond_1/cond_text&batch_normalization_8/cond_1/pred_id:0'batch_normalization_8/cond_1/switch_t:0 *w
$batch_normalization_8/cond_1/Const:0
&batch_normalization_8/cond_1/pred_id:0
'batch_normalization_8/cond_1/switch_t:0
ö
(batch_normalization_8/cond_1/cond_text_1&batch_normalization_8/cond_1/pred_id:0'batch_normalization_8/cond_1/switch_f:0*y
&batch_normalization_8/cond_1/Const_1:0
&batch_normalization_8/cond_1/pred_id:0
'batch_normalization_8/cond_1/switch_f:0
Ü
$batch_normalization_9/cond/cond_text$batch_normalization_9/cond/pred_id:0%batch_normalization_9/cond/switch_t:0 *ä

Conv2D_9:0
!batch_normalization_9/beta/read:0
"batch_normalization_9/cond/Const:0
$batch_normalization_9/cond/Const_1:0
2batch_normalization_9/cond/FusedBatchNorm/Switch:1
4batch_normalization_9/cond/FusedBatchNorm/Switch_1:1
4batch_normalization_9/cond/FusedBatchNorm/Switch_2:1
+batch_normalization_9/cond/FusedBatchNorm:0
+batch_normalization_9/cond/FusedBatchNorm:1
+batch_normalization_9/cond/FusedBatchNorm:2
+batch_normalization_9/cond/FusedBatchNorm:3
+batch_normalization_9/cond/FusedBatchNorm:4
$batch_normalization_9/cond/pred_id:0
%batch_normalization_9/cond/switch_t:0
"batch_normalization_9/gamma/read:0Z
"batch_normalization_9/gamma/read:04batch_normalization_9/cond/FusedBatchNorm/Switch_1:1Y
!batch_normalization_9/beta/read:04batch_normalization_9/cond/FusedBatchNorm/Switch_2:1@

Conv2D_9:02batch_normalization_9/cond/FusedBatchNorm/Switch:1
ź

&batch_normalization_9/cond/cond_text_1$batch_normalization_9/cond/pred_id:0%batch_normalization_9/cond/switch_f:0*Ä	

Conv2D_9:0
!batch_normalization_9/beta/read:0
4batch_normalization_9/cond/FusedBatchNorm_1/Switch:0
6batch_normalization_9/cond/FusedBatchNorm_1/Switch_1:0
6batch_normalization_9/cond/FusedBatchNorm_1/Switch_2:0
6batch_normalization_9/cond/FusedBatchNorm_1/Switch_3:0
6batch_normalization_9/cond/FusedBatchNorm_1/Switch_4:0
-batch_normalization_9/cond/FusedBatchNorm_1:0
-batch_normalization_9/cond/FusedBatchNorm_1:1
-batch_normalization_9/cond/FusedBatchNorm_1:2
-batch_normalization_9/cond/FusedBatchNorm_1:3
-batch_normalization_9/cond/FusedBatchNorm_1:4
$batch_normalization_9/cond/pred_id:0
%batch_normalization_9/cond/switch_f:0
"batch_normalization_9/gamma/read:0
(batch_normalization_9/moving_mean/read:0
,batch_normalization_9/moving_variance/read:0f
,batch_normalization_9/moving_variance/read:06batch_normalization_9/cond/FusedBatchNorm_1/Switch_4:0b
(batch_normalization_9/moving_mean/read:06batch_normalization_9/cond/FusedBatchNorm_1/Switch_3:0\
"batch_normalization_9/gamma/read:06batch_normalization_9/cond/FusedBatchNorm_1/Switch_1:0[
!batch_normalization_9/beta/read:06batch_normalization_9/cond/FusedBatchNorm_1/Switch_2:0B

Conv2D_9:04batch_normalization_9/cond/FusedBatchNorm_1/Switch:0
ô
&batch_normalization_9/cond_1/cond_text&batch_normalization_9/cond_1/pred_id:0'batch_normalization_9/cond_1/switch_t:0 *w
$batch_normalization_9/cond_1/Const:0
&batch_normalization_9/cond_1/pred_id:0
'batch_normalization_9/cond_1/switch_t:0
ö
(batch_normalization_9/cond_1/cond_text_1&batch_normalization_9/cond_1/pred_id:0'batch_normalization_9/cond_1/switch_f:0*y
&batch_normalization_9/cond_1/Const_1:0
&batch_normalization_9/cond_1/pred_id:0
'batch_normalization_9/cond_1/switch_f:0"Ö

update_opsÇ
Ä
%batch_normalization/AssignMovingAvg:0
'batch_normalization/AssignMovingAvg_1:0
'batch_normalization_1/AssignMovingAvg:0
)batch_normalization_1/AssignMovingAvg_1:0
'batch_normalization_2/AssignMovingAvg:0
)batch_normalization_2/AssignMovingAvg_1:0
'batch_normalization_3/AssignMovingAvg:0
)batch_normalization_3/AssignMovingAvg_1:0
'batch_normalization_4/AssignMovingAvg:0
)batch_normalization_4/AssignMovingAvg_1:0
'batch_normalization_5/AssignMovingAvg:0
)batch_normalization_5/AssignMovingAvg_1:0
'batch_normalization_6/AssignMovingAvg:0
)batch_normalization_6/AssignMovingAvg_1:0
'batch_normalization_7/AssignMovingAvg:0
)batch_normalization_7/AssignMovingAvg_1:0
'batch_normalization_8/AssignMovingAvg:0
)batch_normalization_8/AssignMovingAvg_1:0
'batch_normalization_9/AssignMovingAvg:0
)batch_normalization_9/AssignMovingAvg_1:0"Ě%
trainable_variables´%ą%
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02Const:0

batch_normalization/gamma:0 batch_normalization/gamma/Assign batch_normalization/gamma/read:02,batch_normalization/gamma/Initializer/ones:0

batch_normalization/beta:0batch_normalization/beta/Assignbatch_normalization/beta/read:02,batch_normalization/beta/Initializer/zeros:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_1:0

batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:02.batch_normalization_1/gamma/Initializer/ones:0

batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:02.batch_normalization_1/beta/Initializer/zeros:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_2:0

batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:02.batch_normalization_2/gamma/Initializer/ones:0

batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:02.batch_normalization_2/beta/Initializer/zeros:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:0

batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0

batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
J
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_4:0
?
Variable_9:0Variable_9/AssignVariable_9/read:02	Const_4:0

batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign"batch_normalization_4/gamma/read:02.batch_normalization_4/gamma/Initializer/ones:0

batch_normalization_4/beta:0!batch_normalization_4/beta/Assign!batch_normalization_4/beta/read:02.batch_normalization_4/beta/Initializer/zeros:0
M
Variable_10:0Variable_10/AssignVariable_10/read:02truncated_normal_5:0
B
Variable_11:0Variable_11/AssignVariable_11/read:02	Const_5:0

batch_normalization_5/gamma:0"batch_normalization_5/gamma/Assign"batch_normalization_5/gamma/read:02.batch_normalization_5/gamma/Initializer/ones:0

batch_normalization_5/beta:0!batch_normalization_5/beta/Assign!batch_normalization_5/beta/read:02.batch_normalization_5/beta/Initializer/zeros:0
M
Variable_12:0Variable_12/AssignVariable_12/read:02truncated_normal_6:0
B
Variable_13:0Variable_13/AssignVariable_13/read:02	Const_6:0

batch_normalization_6/gamma:0"batch_normalization_6/gamma/Assign"batch_normalization_6/gamma/read:02.batch_normalization_6/gamma/Initializer/ones:0

batch_normalization_6/beta:0!batch_normalization_6/beta/Assign!batch_normalization_6/beta/read:02.batch_normalization_6/beta/Initializer/zeros:0
M
Variable_14:0Variable_14/AssignVariable_14/read:02truncated_normal_7:0
B
Variable_15:0Variable_15/AssignVariable_15/read:02	Const_7:0

batch_normalization_7/gamma:0"batch_normalization_7/gamma/Assign"batch_normalization_7/gamma/read:02.batch_normalization_7/gamma/Initializer/ones:0

batch_normalization_7/beta:0!batch_normalization_7/beta/Assign!batch_normalization_7/beta/read:02.batch_normalization_7/beta/Initializer/zeros:0
M
Variable_16:0Variable_16/AssignVariable_16/read:02truncated_normal_8:0
B
Variable_17:0Variable_17/AssignVariable_17/read:02	Const_8:0

batch_normalization_8/gamma:0"batch_normalization_8/gamma/Assign"batch_normalization_8/gamma/read:02.batch_normalization_8/gamma/Initializer/ones:0

batch_normalization_8/beta:0!batch_normalization_8/beta/Assign!batch_normalization_8/beta/read:02.batch_normalization_8/beta/Initializer/zeros:0
M
Variable_18:0Variable_18/AssignVariable_18/read:02truncated_normal_9:0
B
Variable_19:0Variable_19/AssignVariable_19/read:02	Const_9:0

batch_normalization_9/gamma:0"batch_normalization_9/gamma/Assign"batch_normalization_9/gamma/read:02.batch_normalization_9/gamma/Initializer/ones:0

batch_normalization_9/beta:0!batch_normalization_9/beta/Assign!batch_normalization_9/beta/read:02.batch_normalization_9/beta/Initializer/zeros:0
T
Variable_20:0Variable_20/AssignVariable_20/read:02Variable_20/initial_value:0
T
Variable_21:0Variable_21/AssignVariable_21/read:02Variable_21/initial_value:0
N
Variable_22:0Variable_22/AssignVariable_22/read:02truncated_normal_10:0
C
Variable_23:0Variable_23/AssignVariable_23/read:02
Const_10:0"
	summaries


loss:0"ŕŹ
	variablesŃŹÍŹ
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02Const:0

batch_normalization/gamma:0 batch_normalization/gamma/Assign batch_normalization/gamma/read:02,batch_normalization/gamma/Initializer/ones:0

batch_normalization/beta:0batch_normalization/beta/Assignbatch_normalization/beta/read:02,batch_normalization/beta/Initializer/zeros:0
¨
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign&batch_normalization/moving_mean/read:023batch_normalization/moving_mean/Initializer/zeros:0
ˇ
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign*batch_normalization/moving_variance/read:026batch_normalization/moving_variance/Initializer/ones:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	Const_1:0

batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:02.batch_normalization_1/gamma/Initializer/ones:0

batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:02.batch_normalization_1/beta/Initializer/zeros:0
°
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign(batch_normalization_1/moving_mean/read:025batch_normalization_1/moving_mean/Initializer/zeros:0
ż
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign,batch_normalization_1/moving_variance/read:028batch_normalization_1/moving_variance/Initializer/ones:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	Const_2:0

batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:02.batch_normalization_2/gamma/Initializer/ones:0

batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:02.batch_normalization_2/beta/Initializer/zeros:0
°
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign(batch_normalization_2/moving_mean/read:025batch_normalization_2/moving_mean/Initializer/zeros:0
ż
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign,batch_normalization_2/moving_variance/read:028batch_normalization_2/moving_variance/Initializer/ones:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	Const_3:0

batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0

batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
°
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign(batch_normalization_3/moving_mean/read:025batch_normalization_3/moving_mean/Initializer/zeros:0
ż
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign,batch_normalization_3/moving_variance/read:028batch_normalization_3/moving_variance/Initializer/ones:0
J
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_4:0
?
Variable_9:0Variable_9/AssignVariable_9/read:02	Const_4:0

batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign"batch_normalization_4/gamma/read:02.batch_normalization_4/gamma/Initializer/ones:0

batch_normalization_4/beta:0!batch_normalization_4/beta/Assign!batch_normalization_4/beta/read:02.batch_normalization_4/beta/Initializer/zeros:0
°
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign(batch_normalization_4/moving_mean/read:025batch_normalization_4/moving_mean/Initializer/zeros:0
ż
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign,batch_normalization_4/moving_variance/read:028batch_normalization_4/moving_variance/Initializer/ones:0
M
Variable_10:0Variable_10/AssignVariable_10/read:02truncated_normal_5:0
B
Variable_11:0Variable_11/AssignVariable_11/read:02	Const_5:0

batch_normalization_5/gamma:0"batch_normalization_5/gamma/Assign"batch_normalization_5/gamma/read:02.batch_normalization_5/gamma/Initializer/ones:0

batch_normalization_5/beta:0!batch_normalization_5/beta/Assign!batch_normalization_5/beta/read:02.batch_normalization_5/beta/Initializer/zeros:0
°
#batch_normalization_5/moving_mean:0(batch_normalization_5/moving_mean/Assign(batch_normalization_5/moving_mean/read:025batch_normalization_5/moving_mean/Initializer/zeros:0
ż
'batch_normalization_5/moving_variance:0,batch_normalization_5/moving_variance/Assign,batch_normalization_5/moving_variance/read:028batch_normalization_5/moving_variance/Initializer/ones:0
M
Variable_12:0Variable_12/AssignVariable_12/read:02truncated_normal_6:0
B
Variable_13:0Variable_13/AssignVariable_13/read:02	Const_6:0

batch_normalization_6/gamma:0"batch_normalization_6/gamma/Assign"batch_normalization_6/gamma/read:02.batch_normalization_6/gamma/Initializer/ones:0

batch_normalization_6/beta:0!batch_normalization_6/beta/Assign!batch_normalization_6/beta/read:02.batch_normalization_6/beta/Initializer/zeros:0
°
#batch_normalization_6/moving_mean:0(batch_normalization_6/moving_mean/Assign(batch_normalization_6/moving_mean/read:025batch_normalization_6/moving_mean/Initializer/zeros:0
ż
'batch_normalization_6/moving_variance:0,batch_normalization_6/moving_variance/Assign,batch_normalization_6/moving_variance/read:028batch_normalization_6/moving_variance/Initializer/ones:0
M
Variable_14:0Variable_14/AssignVariable_14/read:02truncated_normal_7:0
B
Variable_15:0Variable_15/AssignVariable_15/read:02	Const_7:0

batch_normalization_7/gamma:0"batch_normalization_7/gamma/Assign"batch_normalization_7/gamma/read:02.batch_normalization_7/gamma/Initializer/ones:0

batch_normalization_7/beta:0!batch_normalization_7/beta/Assign!batch_normalization_7/beta/read:02.batch_normalization_7/beta/Initializer/zeros:0
°
#batch_normalization_7/moving_mean:0(batch_normalization_7/moving_mean/Assign(batch_normalization_7/moving_mean/read:025batch_normalization_7/moving_mean/Initializer/zeros:0
ż
'batch_normalization_7/moving_variance:0,batch_normalization_7/moving_variance/Assign,batch_normalization_7/moving_variance/read:028batch_normalization_7/moving_variance/Initializer/ones:0
M
Variable_16:0Variable_16/AssignVariable_16/read:02truncated_normal_8:0
B
Variable_17:0Variable_17/AssignVariable_17/read:02	Const_8:0

batch_normalization_8/gamma:0"batch_normalization_8/gamma/Assign"batch_normalization_8/gamma/read:02.batch_normalization_8/gamma/Initializer/ones:0

batch_normalization_8/beta:0!batch_normalization_8/beta/Assign!batch_normalization_8/beta/read:02.batch_normalization_8/beta/Initializer/zeros:0
°
#batch_normalization_8/moving_mean:0(batch_normalization_8/moving_mean/Assign(batch_normalization_8/moving_mean/read:025batch_normalization_8/moving_mean/Initializer/zeros:0
ż
'batch_normalization_8/moving_variance:0,batch_normalization_8/moving_variance/Assign,batch_normalization_8/moving_variance/read:028batch_normalization_8/moving_variance/Initializer/ones:0
M
Variable_18:0Variable_18/AssignVariable_18/read:02truncated_normal_9:0
B
Variable_19:0Variable_19/AssignVariable_19/read:02	Const_9:0

batch_normalization_9/gamma:0"batch_normalization_9/gamma/Assign"batch_normalization_9/gamma/read:02.batch_normalization_9/gamma/Initializer/ones:0

batch_normalization_9/beta:0!batch_normalization_9/beta/Assign!batch_normalization_9/beta/read:02.batch_normalization_9/beta/Initializer/zeros:0
°
#batch_normalization_9/moving_mean:0(batch_normalization_9/moving_mean/Assign(batch_normalization_9/moving_mean/read:025batch_normalization_9/moving_mean/Initializer/zeros:0
ż
'batch_normalization_9/moving_variance:0,batch_normalization_9/moving_variance/Assign,batch_normalization_9/moving_variance/read:028batch_normalization_9/moving_variance/Initializer/ones:0
T
Variable_20:0Variable_20/AssignVariable_20/read:02Variable_20/initial_value:0
T
Variable_21:0Variable_21/AssignVariable_21/read:02Variable_21/initial_value:0
N
Variable_22:0Variable_22/AssignVariable_22/read:02truncated_normal_10:0
C
Variable_23:0Variable_23/AssignVariable_23/read:02
Const_10:0
p
Variable/Adadelta:0Variable/Adadelta/AssignVariable/Adadelta/read:02%Variable/Adadelta/Initializer/zeros:0
x
Variable/Adadelta_1:0Variable/Adadelta_1/AssignVariable/Adadelta_1/read:02'Variable/Adadelta_1/Initializer/zeros:0
x
Variable_1/Adadelta:0Variable_1/Adadelta/AssignVariable_1/Adadelta/read:02'Variable_1/Adadelta/Initializer/zeros:0

Variable_1/Adadelta_1:0Variable_1/Adadelta_1/AssignVariable_1/Adadelta_1/read:02)Variable_1/Adadelta_1/Initializer/zeros:0
´
$batch_normalization/gamma/Adadelta:0)batch_normalization/gamma/Adadelta/Assign)batch_normalization/gamma/Adadelta/read:026batch_normalization/gamma/Adadelta/Initializer/zeros:0
ź
&batch_normalization/gamma/Adadelta_1:0+batch_normalization/gamma/Adadelta_1/Assign+batch_normalization/gamma/Adadelta_1/read:028batch_normalization/gamma/Adadelta_1/Initializer/zeros:0
°
#batch_normalization/beta/Adadelta:0(batch_normalization/beta/Adadelta/Assign(batch_normalization/beta/Adadelta/read:025batch_normalization/beta/Adadelta/Initializer/zeros:0
¸
%batch_normalization/beta/Adadelta_1:0*batch_normalization/beta/Adadelta_1/Assign*batch_normalization/beta/Adadelta_1/read:027batch_normalization/beta/Adadelta_1/Initializer/zeros:0
x
Variable_2/Adadelta:0Variable_2/Adadelta/AssignVariable_2/Adadelta/read:02'Variable_2/Adadelta/Initializer/zeros:0

Variable_2/Adadelta_1:0Variable_2/Adadelta_1/AssignVariable_2/Adadelta_1/read:02)Variable_2/Adadelta_1/Initializer/zeros:0
x
Variable_3/Adadelta:0Variable_3/Adadelta/AssignVariable_3/Adadelta/read:02'Variable_3/Adadelta/Initializer/zeros:0

Variable_3/Adadelta_1:0Variable_3/Adadelta_1/AssignVariable_3/Adadelta_1/read:02)Variable_3/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_1/gamma/Adadelta:0+batch_normalization_1/gamma/Adadelta/Assign+batch_normalization_1/gamma/Adadelta/read:028batch_normalization_1/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_1/gamma/Adadelta_1:0-batch_normalization_1/gamma/Adadelta_1/Assign-batch_normalization_1/gamma/Adadelta_1/read:02:batch_normalization_1/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_1/beta/Adadelta:0*batch_normalization_1/beta/Adadelta/Assign*batch_normalization_1/beta/Adadelta/read:027batch_normalization_1/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_1/beta/Adadelta_1:0,batch_normalization_1/beta/Adadelta_1/Assign,batch_normalization_1/beta/Adadelta_1/read:029batch_normalization_1/beta/Adadelta_1/Initializer/zeros:0
x
Variable_4/Adadelta:0Variable_4/Adadelta/AssignVariable_4/Adadelta/read:02'Variable_4/Adadelta/Initializer/zeros:0

Variable_4/Adadelta_1:0Variable_4/Adadelta_1/AssignVariable_4/Adadelta_1/read:02)Variable_4/Adadelta_1/Initializer/zeros:0
x
Variable_5/Adadelta:0Variable_5/Adadelta/AssignVariable_5/Adadelta/read:02'Variable_5/Adadelta/Initializer/zeros:0

Variable_5/Adadelta_1:0Variable_5/Adadelta_1/AssignVariable_5/Adadelta_1/read:02)Variable_5/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_2/gamma/Adadelta:0+batch_normalization_2/gamma/Adadelta/Assign+batch_normalization_2/gamma/Adadelta/read:028batch_normalization_2/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_2/gamma/Adadelta_1:0-batch_normalization_2/gamma/Adadelta_1/Assign-batch_normalization_2/gamma/Adadelta_1/read:02:batch_normalization_2/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_2/beta/Adadelta:0*batch_normalization_2/beta/Adadelta/Assign*batch_normalization_2/beta/Adadelta/read:027batch_normalization_2/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_2/beta/Adadelta_1:0,batch_normalization_2/beta/Adadelta_1/Assign,batch_normalization_2/beta/Adadelta_1/read:029batch_normalization_2/beta/Adadelta_1/Initializer/zeros:0
x
Variable_6/Adadelta:0Variable_6/Adadelta/AssignVariable_6/Adadelta/read:02'Variable_6/Adadelta/Initializer/zeros:0

Variable_6/Adadelta_1:0Variable_6/Adadelta_1/AssignVariable_6/Adadelta_1/read:02)Variable_6/Adadelta_1/Initializer/zeros:0
x
Variable_7/Adadelta:0Variable_7/Adadelta/AssignVariable_7/Adadelta/read:02'Variable_7/Adadelta/Initializer/zeros:0

Variable_7/Adadelta_1:0Variable_7/Adadelta_1/AssignVariable_7/Adadelta_1/read:02)Variable_7/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_3/gamma/Adadelta:0+batch_normalization_3/gamma/Adadelta/Assign+batch_normalization_3/gamma/Adadelta/read:028batch_normalization_3/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_3/gamma/Adadelta_1:0-batch_normalization_3/gamma/Adadelta_1/Assign-batch_normalization_3/gamma/Adadelta_1/read:02:batch_normalization_3/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_3/beta/Adadelta:0*batch_normalization_3/beta/Adadelta/Assign*batch_normalization_3/beta/Adadelta/read:027batch_normalization_3/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_3/beta/Adadelta_1:0,batch_normalization_3/beta/Adadelta_1/Assign,batch_normalization_3/beta/Adadelta_1/read:029batch_normalization_3/beta/Adadelta_1/Initializer/zeros:0
x
Variable_8/Adadelta:0Variable_8/Adadelta/AssignVariable_8/Adadelta/read:02'Variable_8/Adadelta/Initializer/zeros:0

Variable_8/Adadelta_1:0Variable_8/Adadelta_1/AssignVariable_8/Adadelta_1/read:02)Variable_8/Adadelta_1/Initializer/zeros:0
x
Variable_9/Adadelta:0Variable_9/Adadelta/AssignVariable_9/Adadelta/read:02'Variable_9/Adadelta/Initializer/zeros:0

Variable_9/Adadelta_1:0Variable_9/Adadelta_1/AssignVariable_9/Adadelta_1/read:02)Variable_9/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_4/gamma/Adadelta:0+batch_normalization_4/gamma/Adadelta/Assign+batch_normalization_4/gamma/Adadelta/read:028batch_normalization_4/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_4/gamma/Adadelta_1:0-batch_normalization_4/gamma/Adadelta_1/Assign-batch_normalization_4/gamma/Adadelta_1/read:02:batch_normalization_4/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_4/beta/Adadelta:0*batch_normalization_4/beta/Adadelta/Assign*batch_normalization_4/beta/Adadelta/read:027batch_normalization_4/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_4/beta/Adadelta_1:0,batch_normalization_4/beta/Adadelta_1/Assign,batch_normalization_4/beta/Adadelta_1/read:029batch_normalization_4/beta/Adadelta_1/Initializer/zeros:0
|
Variable_10/Adadelta:0Variable_10/Adadelta/AssignVariable_10/Adadelta/read:02(Variable_10/Adadelta/Initializer/zeros:0

Variable_10/Adadelta_1:0Variable_10/Adadelta_1/AssignVariable_10/Adadelta_1/read:02*Variable_10/Adadelta_1/Initializer/zeros:0
|
Variable_11/Adadelta:0Variable_11/Adadelta/AssignVariable_11/Adadelta/read:02(Variable_11/Adadelta/Initializer/zeros:0

Variable_11/Adadelta_1:0Variable_11/Adadelta_1/AssignVariable_11/Adadelta_1/read:02*Variable_11/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_5/gamma/Adadelta:0+batch_normalization_5/gamma/Adadelta/Assign+batch_normalization_5/gamma/Adadelta/read:028batch_normalization_5/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_5/gamma/Adadelta_1:0-batch_normalization_5/gamma/Adadelta_1/Assign-batch_normalization_5/gamma/Adadelta_1/read:02:batch_normalization_5/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_5/beta/Adadelta:0*batch_normalization_5/beta/Adadelta/Assign*batch_normalization_5/beta/Adadelta/read:027batch_normalization_5/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_5/beta/Adadelta_1:0,batch_normalization_5/beta/Adadelta_1/Assign,batch_normalization_5/beta/Adadelta_1/read:029batch_normalization_5/beta/Adadelta_1/Initializer/zeros:0
|
Variable_12/Adadelta:0Variable_12/Adadelta/AssignVariable_12/Adadelta/read:02(Variable_12/Adadelta/Initializer/zeros:0

Variable_12/Adadelta_1:0Variable_12/Adadelta_1/AssignVariable_12/Adadelta_1/read:02*Variable_12/Adadelta_1/Initializer/zeros:0
|
Variable_13/Adadelta:0Variable_13/Adadelta/AssignVariable_13/Adadelta/read:02(Variable_13/Adadelta/Initializer/zeros:0

Variable_13/Adadelta_1:0Variable_13/Adadelta_1/AssignVariable_13/Adadelta_1/read:02*Variable_13/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_6/gamma/Adadelta:0+batch_normalization_6/gamma/Adadelta/Assign+batch_normalization_6/gamma/Adadelta/read:028batch_normalization_6/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_6/gamma/Adadelta_1:0-batch_normalization_6/gamma/Adadelta_1/Assign-batch_normalization_6/gamma/Adadelta_1/read:02:batch_normalization_6/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_6/beta/Adadelta:0*batch_normalization_6/beta/Adadelta/Assign*batch_normalization_6/beta/Adadelta/read:027batch_normalization_6/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_6/beta/Adadelta_1:0,batch_normalization_6/beta/Adadelta_1/Assign,batch_normalization_6/beta/Adadelta_1/read:029batch_normalization_6/beta/Adadelta_1/Initializer/zeros:0
|
Variable_14/Adadelta:0Variable_14/Adadelta/AssignVariable_14/Adadelta/read:02(Variable_14/Adadelta/Initializer/zeros:0

Variable_14/Adadelta_1:0Variable_14/Adadelta_1/AssignVariable_14/Adadelta_1/read:02*Variable_14/Adadelta_1/Initializer/zeros:0
|
Variable_15/Adadelta:0Variable_15/Adadelta/AssignVariable_15/Adadelta/read:02(Variable_15/Adadelta/Initializer/zeros:0

Variable_15/Adadelta_1:0Variable_15/Adadelta_1/AssignVariable_15/Adadelta_1/read:02*Variable_15/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_7/gamma/Adadelta:0+batch_normalization_7/gamma/Adadelta/Assign+batch_normalization_7/gamma/Adadelta/read:028batch_normalization_7/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_7/gamma/Adadelta_1:0-batch_normalization_7/gamma/Adadelta_1/Assign-batch_normalization_7/gamma/Adadelta_1/read:02:batch_normalization_7/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_7/beta/Adadelta:0*batch_normalization_7/beta/Adadelta/Assign*batch_normalization_7/beta/Adadelta/read:027batch_normalization_7/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_7/beta/Adadelta_1:0,batch_normalization_7/beta/Adadelta_1/Assign,batch_normalization_7/beta/Adadelta_1/read:029batch_normalization_7/beta/Adadelta_1/Initializer/zeros:0
|
Variable_16/Adadelta:0Variable_16/Adadelta/AssignVariable_16/Adadelta/read:02(Variable_16/Adadelta/Initializer/zeros:0

Variable_16/Adadelta_1:0Variable_16/Adadelta_1/AssignVariable_16/Adadelta_1/read:02*Variable_16/Adadelta_1/Initializer/zeros:0
|
Variable_17/Adadelta:0Variable_17/Adadelta/AssignVariable_17/Adadelta/read:02(Variable_17/Adadelta/Initializer/zeros:0

Variable_17/Adadelta_1:0Variable_17/Adadelta_1/AssignVariable_17/Adadelta_1/read:02*Variable_17/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_8/gamma/Adadelta:0+batch_normalization_8/gamma/Adadelta/Assign+batch_normalization_8/gamma/Adadelta/read:028batch_normalization_8/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_8/gamma/Adadelta_1:0-batch_normalization_8/gamma/Adadelta_1/Assign-batch_normalization_8/gamma/Adadelta_1/read:02:batch_normalization_8/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_8/beta/Adadelta:0*batch_normalization_8/beta/Adadelta/Assign*batch_normalization_8/beta/Adadelta/read:027batch_normalization_8/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_8/beta/Adadelta_1:0,batch_normalization_8/beta/Adadelta_1/Assign,batch_normalization_8/beta/Adadelta_1/read:029batch_normalization_8/beta/Adadelta_1/Initializer/zeros:0
|
Variable_18/Adadelta:0Variable_18/Adadelta/AssignVariable_18/Adadelta/read:02(Variable_18/Adadelta/Initializer/zeros:0

Variable_18/Adadelta_1:0Variable_18/Adadelta_1/AssignVariable_18/Adadelta_1/read:02*Variable_18/Adadelta_1/Initializer/zeros:0
|
Variable_19/Adadelta:0Variable_19/Adadelta/AssignVariable_19/Adadelta/read:02(Variable_19/Adadelta/Initializer/zeros:0

Variable_19/Adadelta_1:0Variable_19/Adadelta_1/AssignVariable_19/Adadelta_1/read:02*Variable_19/Adadelta_1/Initializer/zeros:0
ź
&batch_normalization_9/gamma/Adadelta:0+batch_normalization_9/gamma/Adadelta/Assign+batch_normalization_9/gamma/Adadelta/read:028batch_normalization_9/gamma/Adadelta/Initializer/zeros:0
Ä
(batch_normalization_9/gamma/Adadelta_1:0-batch_normalization_9/gamma/Adadelta_1/Assign-batch_normalization_9/gamma/Adadelta_1/read:02:batch_normalization_9/gamma/Adadelta_1/Initializer/zeros:0
¸
%batch_normalization_9/beta/Adadelta:0*batch_normalization_9/beta/Adadelta/Assign*batch_normalization_9/beta/Adadelta/read:027batch_normalization_9/beta/Adadelta/Initializer/zeros:0
Ŕ
'batch_normalization_9/beta/Adadelta_1:0,batch_normalization_9/beta/Adadelta_1/Assign,batch_normalization_9/beta/Adadelta_1/read:029batch_normalization_9/beta/Adadelta_1/Initializer/zeros:0
|
Variable_20/Adadelta:0Variable_20/Adadelta/AssignVariable_20/Adadelta/read:02(Variable_20/Adadelta/Initializer/zeros:0

Variable_20/Adadelta_1:0Variable_20/Adadelta_1/AssignVariable_20/Adadelta_1/read:02*Variable_20/Adadelta_1/Initializer/zeros:0
|
Variable_21/Adadelta:0Variable_21/Adadelta/AssignVariable_21/Adadelta/read:02(Variable_21/Adadelta/Initializer/zeros:0

Variable_21/Adadelta_1:0Variable_21/Adadelta_1/AssignVariable_21/Adadelta_1/read:02*Variable_21/Adadelta_1/Initializer/zeros:0
|
Variable_22/Adadelta:0Variable_22/Adadelta/AssignVariable_22/Adadelta/read:02(Variable_22/Adadelta/Initializer/zeros:0

Variable_22/Adadelta_1:0Variable_22/Adadelta_1/AssignVariable_22/Adadelta_1/read:02*Variable_22/Adadelta_1/Initializer/zeros:00L¨       ŁK"	Á§Ž§ŻÖA*

lossWŁčG	ô       Ř-	B
I!§ŻÖA*

lossłçGľú%Ŕ       Ř-	ŐŠÖ#§ŻÖA*

lossłçGĆ˝Â        Ř-	Tva&§ŻÖA*

lossČ$ęGUaĹ       Ř-	7ň(§ŻÖA*

lossÂrçGűçf5       Ř-	{+§ŻÖA*

loss(äGÜ!¸       Ř-	ŞÖ.§ŻÖA*

losskúęGëËL       Ř-	-0§ŻÖA*

loss,§ćG˝îNŃ       Ř-	ő)3§ŻÖA*

loss?ăGŹCĎ       Ř-	ÄÁ5§ŻÖA	*

loss˝¸ăG:rč       Ř-	ĚŢM8§ŻÖA
*

lossĹăGy|       Ř-	&Ţ:§ŻÖA*

lossOäGĂe*       Ř-	úłn=§ŻÖA*

loss7ćGhd´÷       Ř-	Č@§ŻÖA*

losscĽăGLř       Ř-	wB§ŻÖA*

loss%ßGţv       Ř-	´§!E§ŻÖA*

lossĽçGBĺ4       Ř-	y˛G§ŻÖA*

loss´ćGVmGő       Ř-	PJ§ŻÖA*

lossxĺGz˘jž       Ř-	v[áL§ŻÖA*

lossßSĺGćśÁ
       Ř-	ŚmO§ŻÖA*

loss}gâGb^¸       Ř-	ń˙Q§ŻÖA*

loss÷ŽćG[âÂ       Ř-	ú T§ŻÖA*

lossÓĺGá       Ř-	Uí=W§ŻÖA*

lossmbčGJj       Ř-	}ŐY§ŻÖA*

loss.hŕG Š       Ř-	\ßh\§ŻÖA*

lossö]×GZËYp       Ř-	­i_§ŻÖA*

lossŃÜG˙mx       Ř-	ŕśa§ŻÖA*

lossŽ4ÝGîŘ       Ř-	ë¤-d§ŻÖA*

loss¤ÝGxąó×       Ř-	śáĂf§ŻÖA*

lossIôÚGHĆ-       Ř-	ćîki§ŻÖA*

loss#ĺG9ä<       Ř-	ćGl§ŻÖA*

lossďŰGţîvÍ       Ř-	Žn§ŻÖA*

lossČŕGw)@       Ř-	-q§ŻÖA *

losscQăGÖřőú       Ř-	ĂóÚs§ŻÖA!*

loss>öŕGs÷i       Ř-	wńqv§ŻÖA"*

loss ćŰGFke       Ř-	P y§ŻÖA#*

lossŐGc3Ďv       Ř-	Ďđ{§ŻÖA$*

lossňÚGr6ń       Ř-	¤9E~§ŻÖA%*

loss(âŰGx:Q       Ř-	ÔŹŢ§ŻÖA&*

lossEĚÚGŔÝ       Ř-	av§ŻÖA'*

lossP¨×GĆ'?       Ř-	=c
§ŻÖA(*

lossńŰG:Ćrˇ       Ř-	zŚś§ŻÖA)*

lossËÇßGĂ)ş       Ř-	CUN§ŻÖA**

lossdĽÔGO ŕ       Ř-	óă§ŻÖA+*

lossâŮGŽ¨       Ř-	u}y§ŻÖA,*

lossj^ŐGţV*       Ř-	Ůá&§ŻÖA-*

loss>^ŇGţ¸¤       Ř-	i(Â§ŻÖA.*

loss~ÓGÍÉ7Ź       Ř-	m\T§ŻÖA/*

lossĹÖÔGňm       Ř-	$Ůç§ŻÖA0*

lossâŕŰGČQOf       Ř-	>á§ŻÖA1*

lossÜGřč?č       Ř-	+21 §ŻÖA2*

losse;ÜG$×ˇ       Ř-	ĘĆ˘§ŻÖA3*

loss3]ÖGËs´       Ř-	XĽ§ŻÖA4*

lossś5×G|đ°l       Ř-	ř­¨§ŻÖA5*

lossv$ÔGlú"       Ř-	@đ¨Ş§ŻÖA6*

lossćHŇGŚ       Ř-	Ł=­§ŻÖA7*

loss ÇŐGaő*       Ř-	ÎŻ§ŻÖA8*

loss¨ĘG}Ĺ5       Ř-	Ś|˛§ŻÖA9*

lossÔGíńö       Ř-	Ľ/ľ§ŻÖA:*

lossâ*ŐG ůČ       Ř-	XęŻˇ§ŻÖA;*

lossĎÔGĺ˝ë.       Ř-	DĽKş§ŻÖA<*

loss~ĐGűPOŮ       Ř-	ŚK ˝§ŻÖA=*

lossÁëÔG(	űN       Ř-	{fż§ŻÖA>*

loss´žŐGŚÖł       Ř-	5u)Â§ŻÖA?*

loss˘FÓGź4!       Ř-	ÓÎÁÄ§ŻÖA@*

lossŮčÔG`ü/\       Ř-	uÇ§ŻÖAA*

loss1ŰGnhYî       Ř-	"Ę§ŻÖAB*

lossť)ŃG4s,       Ř-	ąQŚĚ§ŻÖAC*

loss ÓGG       Ř-	x>Ď§ŻÖAD*

loss÷>×G¨b\       Ř-	SĘ÷Ń§ŻÖAE*

lossćÓGÍi       Ř-	ŤÔ§ŻÖAF*

loss	ąÔGoŤ7       Ř-	Ę>×§ŻÖAG*

loss&DŃGíüN       Ř-	ł ÝŮ§ŻÖAH*

lossňŹŃGýÄ+˘       Ř-	ůÜ§ŻÖAI*

loss ĽÍG5)ĂD       Ř-	ů3ß§ŻÖAJ*

lossŰ5ŃG<       Ř-	ĹéÍá§ŻÖAK*

lossJ×GK