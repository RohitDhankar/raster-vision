from typing import TYPE_CHECKING, List, Optional, Union
from uuid import uuid4

from rastervision.pipeline import rv_config
from rastervision.core.data.utils import listify_uris, get_polygons_from_uris

if TYPE_CHECKING:
    from rastervision.core.data import ClassConfig, Scene


def make_ss_scene(class_config: 'ClassConfig',
                  image_uri: Union[str, List[str]],
                  label_raster_uri: Optional[Union[str, List[str]]] = None,
                  label_vector_uri: Optional[str] = None,
                  aoi_uri: Union[str, List[str]] = [],
                  label_vector_default_class_id: Optional[int] = None,
                  image_raster_source_kw: dict = {},
                  label_raster_source_kw: dict = {},
                  label_vector_source_kw: dict = {}) -> 'Scene':
    """Create a semantic segmentation scene from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        class_config (ClassConfig): The ClassConfig.
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_raster_uri (Optional[Union[str, List[str]]], optional): URI or
            list of URIs of GeoTIFFs to use as the source of segmentation label
            data. If the labels are in the form of GeoJSONs, use
            label_vector_uri instead. Defaults to None.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. If the labels are in the
            form of GeoTIFFs, use label_raster_uri instead. Defaults to None.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polgons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for label data, if label_raster_uri is
            used. See docs for RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSource used for label data, if label_vector_uri
            is used. See docs for GeoJSONVectorSource for more details.
            Defaults to {}.

    Raises:
        ValueError: If both label_raster_uri and label_vector_uri are
            specified.

    Returns:
        Scene: A semantic segmentation scene.
    """
    # use local imports to avoid circular import problems
    from rastervision.core.data import (
        GeoJSONVectorSource, RasterioSource, RasterizedSource, Scene,
        SemanticSegmentationLabelSource, ClassInferenceTransformer)

    if label_raster_uri is not None and label_vector_uri is not None:
        raise ValueError('Specify either label_raster_uri or '
                         'label_vector_uri or neither, but not both.')

    class_config.ensure_null_class()

    image_uri = listify_uris(image_uri)
    raster_source = RasterioSource(uris=image_uri, **image_raster_source_kw)

    crs_transformer = raster_source.get_crs_transformer()
    extent = raster_source.extent
    null_class_id = class_config.get_null_class_id()

    label_raster_source = None
    if label_raster_uri is not None:
        label_raster_uri = listify_uris(label_raster_uri)
        label_raster_source = RasterioSource(
            uris=label_raster_uri, **label_raster_source_kw)
    elif label_vector_uri is not None:
        if label_vector_default_class_id is not None:
            # add a ClassInferenceTransformer to the VectorSource
            class_inf_tf = ClassInferenceTransformer(
                default_class_id=label_vector_default_class_id)
            vector_tfs = label_vector_source_kw.get('vector_transformers', [])
            label_vector_source_kw['vector_transformers'] = (
                [class_inf_tf] + vector_tfs)
        vector_source = GeoJSONVectorSource(
            uri=label_vector_uri,
            ignore_crs_field=True,
            crs_transformer=crs_transformer,
            **label_vector_source_kw)
        label_raster_source = RasterizedSource(
            vector_source=vector_source,
            background_class_id=label_raster_source_kw.pop(
                'background_class_id', null_class_id),
            extent=extent,
            **label_raster_source_kw)

    label_source = None
    if label_raster_source is not None:
        label_source = SemanticSegmentationLabelSource(
            raster_source=label_raster_source, null_class_id=null_class_id)

    aoi_polygons = get_polygons_from_uris(aoi_uri, crs_transformer)
    scene = Scene(
        id=uuid4(),
        raster_source=raster_source,
        label_source=label_source,
        aoi_polygons=aoi_polygons)

    return scene


def make_cc_scene(class_config: 'ClassConfig',
                  image_uri: Union[str, List[str]],
                  label_vector_uri: Optional[str] = None,
                  aoi_uri: Union[str, List[str]] = [],
                  label_vector_default_class_id: Optional[int] = None,
                  image_raster_source_kw: dict = {},
                  label_vector_source_kw: dict = {},
                  label_source_kw: dict = {}) -> 'Scene':
    """Create a chip classification scene from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        class_config (ClassConfig): The ClassConfig.
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. Defaults to None.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polgons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSourceConfig used for label data, if
            label_vector_uri is set. See docs for GeoJSONVectorSourceConfig
            for more details. Defaults to {}.
        label_source_kw (dict, optional): Additional arguments to pass
            to the ChipClassificationLabelSourceConfig used for label data, if
            label_vector_uri is set. See docs for
            ChipClassificationLabelSourceConfig for more details.
            Defaults to {}.
        **kwargs: All other keyword args are passed to the default constructor
            for this class.

    Returns:
        Scene: A chip classification scene.
    """
    # use local imports to avoid circular import problems
    from rastervision.core.data import (
        RasterioSource, Scene, ClassInferenceTransformerConfig,
        ChipClassificationLabelSourceConfig, GeoJSONVectorSourceConfig)

    image_uri = listify_uris(image_uri)
    raster_source = RasterioSource(image_uri, **image_raster_source_kw)

    crs_transformer = raster_source.get_crs_transformer()
    extent = raster_source.extent

    label_source = None
    if label_vector_uri is not None:
        if label_vector_default_class_id is not None:
            # add a ClassInferenceTransformer to the VectorSource
            class_inf_tf = ClassInferenceTransformerConfig(
                default_class_id=label_vector_default_class_id)
            vector_tfs = label_vector_source_kw.get('transformers', [])
            label_vector_source_kw['transformers'] = (
                [class_inf_tf] + vector_tfs)
        geojson_cfg = GeoJSONVectorSourceConfig(
            uri=label_vector_uri,
            ignore_crs_field=True,
            **label_vector_source_kw)
        # use config to ensure required transformers are auto added
        label_source_cfg = ChipClassificationLabelSourceConfig(
            vector_source=geojson_cfg, **label_source_kw)
        label_source = label_source_cfg.build(
            class_config,
            crs_transformer,
            extent=extent,
            tmp_dir=rv_config.get_tmp_dir())

    aoi_polygons = get_polygons_from_uris(aoi_uri, crs_transformer)
    scene = Scene(
        id=uuid4(),
        raster_source=raster_source,
        label_source=label_source,
        aoi_polygons=aoi_polygons)

    return scene


def make_od_scene(class_config: 'ClassConfig',
                  image_uri: Union[str, List[str]],
                  label_vector_uri: Optional[str] = None,
                  aoi_uri: Union[str, List[str]] = [],
                  label_vector_default_class_id: Optional[int] = None,
                  image_raster_source_kw: dict = {},
                  label_vector_source_kw: dict = {},
                  label_source_kw: dict = {}) -> 'Scene':
    """Create an object detection scene from image and label URIs.

    This is a convenience method. For more fine-grained control, it is
    recommended to use the default constructor.

    Args:
        class_config (ClassConfig): The ClassConfig.
        image_uri (Union[str, List[str]]): URI or list of URIs of GeoTIFFs to
            use as the source of image data.
        label_vector_uri (Optional[str], optional):  URI of GeoJSON file to use
            as the source of segmentation label data. Defaults to None.
        aoi_uri (Union[str, List[str]], optional): URI or list of URIs of
            GeoJSONs that specify the area-of-interest. If provided, the
            dataset will only access data from this area. Defaults to [].
        label_vector_default_class_id (Optional[int], optional): If using
            label_vector_uri and all polygons in that file belong to the same
            class and they do not contain a `class_id` property, then use this
            argument to map all of the polgons to the appropriate class ID.
            See docs for ClassInferenceTransformer for more details.
            Defaults to None.
        image_raster_source_kw (dict, optional): Additional arguments to pass
            to the RasterioSource used for image data. See docs for
            RasterioSource for more details. Defaults to {}.
        label_vector_source_kw (dict, optional): Additional arguments to pass
            to the GeoJSONVectorSourceConfig used for label data, if
            label_vector_uri is set. See docs for GeoJSONVectorSourceConfig
            for more details. Defaults to {}.
        label_source_kw (dict, optional): Additional arguments to pass
            to the ObjectDetectionLabelSourceConfig used for label data, if
            label_vector_uri is set. See docs for
            ObjectDetectionLabelSourceConfig for more details.
            Defaults to {}.
        **kwargs: All other keyword args are passed to the default constructor
            for this class.

    Returns:
        Scene: An object detection scene.
    """
    # use local imports to avoid circular import problems
    from rastervision.core.data import (
        RasterioSource, Scene, ClassInferenceTransformerConfig,
        GeoJSONVectorSourceConfig, ObjectDetectionLabelSourceConfig)

    image_uri = listify_uris(image_uri)
    raster_source = RasterioSource(image_uri, **image_raster_source_kw)

    crs_transformer = raster_source.get_crs_transformer()
    extent = raster_source.extent

    label_source = None
    if label_vector_uri is not None:
        if label_vector_default_class_id is not None:
            # add a ClassInferenceTransformer to the VectorSource
            class_inf_tf = ClassInferenceTransformerConfig(
                default_class_id=label_vector_default_class_id)
            vector_tfs = label_vector_source_kw.get('transformers', [])
            label_vector_source_kw['transformers'] = (
                [class_inf_tf] + vector_tfs)
        geojson_cfg = GeoJSONVectorSourceConfig(
            uri=label_vector_uri,
            ignore_crs_field=True,
            **label_vector_source_kw)
        # use config to ensure required transformers are auto added
        label_source_cfg = ObjectDetectionLabelSourceConfig(
            vector_source=geojson_cfg, **label_source_kw)
        label_source = label_source_cfg.build(
            class_config,
            crs_transformer,
            extent=extent,
            tmp_dir=rv_config.get_tmp_dir())

    aoi_polygons = get_polygons_from_uris(aoi_uri, crs_transformer)
    scene = Scene(
        id=uuid4(),
        raster_source=raster_source,
        label_source=label_source,
        aoi_polygons=aoi_polygons)

    return scene
