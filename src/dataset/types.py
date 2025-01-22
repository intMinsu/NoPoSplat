from typing import Callable, Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class BatchedViews(TypedDict, total=False):
    """
        A typed dictionary defining the structure of batched view data for multi-view processing.

        Attributes:
            extrinsics (Float[Tensor, "batch _ 4 4"], optional):
                A tensor containing the extrinsic camera parameters for each view in the batch.
                The shape `[batch, view, 4, 4]` represents the transformation matrix for each view,
                mapping 3D world coordinates to camera coordinates.
            intrinsics (Float[Tensor, "batch _ 3 3"], optional):
                A tensor with intrinsic camera parameters for each view. The shape `[batch, view, 3, 3]`
                defines the calibration matrix used for projecting 3D points into 2D image space.
            image (Float[Tensor, "batch _ _ _ _"], optional):
                The image data for the batch. The shape `[batch, view, channel, height, width]`
                contains pixel values, where `channel` typically represents RGB or grayscale.
            near (Float[Tensor, "batch _"], optional):
                The near plane distance for each view in the batch. The shape `[batch, view]`
                specifies the minimum depth distance for 3D rendering or view analysis.
            far (Float[Tensor, "batch _"], optional):
                The far plane distance for each view in the batch. The shape `[batch, view]`
                defines the maximum depth distance for 3D rendering or view analysis.
            index (Int64[Tensor, "batch _"], optional):
                An integer tensor specifying indices or identifiers for each view in the batch.
                The shape `[batch, view]` allows for mapping views to metadata or additional properties.
            overlap (Float[Tensor, "batch _"], optional):
                A tensor representing the overlap ratio between views in the batch.
                The shape `[batch, view]` quantifies the shared field of view between adjacent or
                related views.

        Example:
            >>> views = BatchedViews(
            ...     extrinsics=torch.rand((4, 2, 4, 4)),  # 4 batches, 2 views, 4x4 matrix
            ...     intrinsics=torch.rand((4, 2, 3, 3)),  # 4 batches, 2 views, 3x3 matrix
            ...     image=torch.rand((4, 2, 3, 128, 128)),  # 4 batches, 2 views, RGB 128x128
            ...     near=torch.rand((4, 2)),  # 4 batches, 2 views
            ...     far=torch.rand((4, 2)),  # 4 batches, 2 views
            ...     index=torch.randint(0, 100, (4, 2)),  # 4 batches, 2 views
            ...     overlap=torch.rand((4, 2))  # 4 batches, 2 views
            ... )
        """
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    overlap: Float[Tensor, "batch _"]  # batch view


class BatchedExample(TypedDict, total=False):
    """
      A typed dictionary representing a batched example.

      Attributes:
          target (BatchedViews, optional):
              Contains the target data for the batch. The specific structure is defined by the `BatchedViews` type.
          context (BatchedViews, optional):
              Represents the contextual data for the batch.  The specific structure is defined by the `BatchedViews` type.
          scene (list[str], optional):
              A list of strings describing scenes associated with the batch.
      Example:
          >>> target_views = BatchedViews(
          ...   extrinsics=torch.rand((4, 2, 4, 4)),  # 4 batches, 2 views, 4x4 matrix
          ...   intrinsics=torch.rand((4, 2, 3, 3)),  # 4 batches, 2 views, 3x3 matrix
          ...   image=torch.rand((4, 2, 3, 128, 128)),  # 4 batches, 2 views, RGB 128x128
          ...   )
          ... batch = BatchedExample(
          ...     target=target_views,
          ...     context=context_views,
          ...     scene=["indoor", "kitchen"]
          ... )
      """
    target: BatchedViews
    context: BatchedViews
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews
