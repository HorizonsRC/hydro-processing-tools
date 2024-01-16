"""DataSourceBlob Object."""
from xml.etree import ElementTree

import pandas as pd
from defusedxml import ElementTree as DefusedElementTree


class ItemInfo:
    """Item Info Class."""

    def __init__(
        self,
        item_number: int,
        item_name: str,
        item_format: str,
        divisor: str,
        units: str,
        format: str,
    ):
        """
        Initialize an ItemInfo instance.

        Parameters
        ----------
        item_number : int
            The item number associated with the item information.
        item_name : str
            The name of the item.
        item_format : str
            The format of the item.
        divisor : str
            The divisor associated with the item.
        units : str
            The units of measurement for the item.
        format : str
            The format of the item information.

        Returns
        -------
        None
        """
        self.item_number = item_number
        self.item_name = item_name
        self.item_format = item_format
        self.divisor = divisor
        self.units = units
        self.format = format

    @classmethod
    def from_xml(cls, source):
        """
        Create an ItemInfo instance from XML.

        Parameters
        ----------
        source : str or ElementTree.Element or bytes or bytearray or file-like object
            The XML source to parse and create an ItemInfo instance.

        Returns
        -------
        ItemInfo
            An instance of ItemInfo created from the XML source.

        Raises
        ------
        ValueError
            If the XML source type is not supported or if the XML structure is invalid.

        Notes
        -----
        This class method reads an object from XML, extracting information to create
        an ItemInfo instance.

        Examples
        --------
        >>> xml_string = "<ItemInfo ItemNumber='1'><ItemName>Example</ItemName></ItemInfo>"
        >>> item_info_instance = ItemInfo.from_xml(xml_string)
        >>> isinstance(item_info_instance, ItemInfo)
        True
        """
        if isinstance(source, str):
            # If the source is a string, treat it as raw XML
            root = DefusedElementTree.fromstring(source)
        elif isinstance(source, ElementTree.Element):
            # If the source is an ElementTree object, use it directly
            root = source
        elif isinstance(source, (bytes, bytearray)):
            # If the source is a bytes or bytearray, assume it's
            # XML content and decode it.
            root = DefusedElementTree.fromstring(source.decode())
        elif hasattr(source, "read"):
            # If the source has a 'read' method, treat it as a
            # file-like object.
            root = DefusedElementTree.parse(source).getroot()
        else:
            raise ValueError("Unsupported XML source type.")

        if root.tag == "ItemInfo":
            item_number = int(root.attrib["ItemNumber"])
        else:
            raise ValueError(
                "Tag at ItemInfo level should be 'ItemInfo'," f" found {root.tag}."
            )

        item_name = str(root.findtext("ItemName"))
        item_format = str(root.findtext("ItemFormat"))
        divisor = str(root.findtext("Divisor"))
        units = str(root.findtext("Units"))
        format = str(root.findtext("Format"))

        return cls(item_number, item_name, item_format, divisor, units, format)

    def to_xml_tree(self):
        """
        Convert the ItemInfo instance to an XML ElementTree.

        Returns
        -------
        ElementTree.Element
            The XML ElementTree representing the ItemInfo instance.

        Notes
        -----
        This method converts the ItemInfo object into an XML blob
        using ElementTree.

        Examples
        --------
        >>> item_info_instance = ItemInfo(1, "Example", "Format", "1", "Units", "Format")
        >>> xml_tree = item_info_instance.to_xml_tree()
        >>> isinstance(xml_tree, ElementTree.Element)
        True
        """
        item_info_root = ElementTree.Element(
            "ItemInfo", attrib={"ItemNumber": str(self.item_number)}
        )

        item_name_element = ElementTree.SubElement(item_info_root, "ItemName")
        item_name_element.text = self.item_name

        item_format_element = ElementTree.SubElement(item_info_root, "ItemFormat")
        item_format_element.text = self.item_format

        divisor_element = ElementTree.SubElement(item_info_root, "Divisor")
        divisor_element.text = self.divisor

        units_element = ElementTree.SubElement(item_info_root, "Units")
        units_element.text = self.units

        format_element = ElementTree.SubElement(item_info_root, "Format")
        format_element.text = self.format

        return item_info_root


class DataSource:
    """Data Source class."""

    def __init__(
        self,
        name: str,
        num_items: int,
        ts_type: str,
        data_type: str,
        interpolation: str,
        item_format: str,
        item_info: list[ItemInfo] | None,
    ):
        """
        Initialize a DataSource instance.

        Parameters
        ----------
        name : str
            The name of the data source.
        num_items : int
            The number of items in the data source.
        ts_type : str
            The time series type of the data source.
        data_type : str
            The data type of the data source.
        interpolation : str
            The interpolation method used by the data source.
        item_format : str
            The format of the items in the data source.
        item_info : list of ItemInfo or None, optional
            A list of ItemInfo objects providing additional information about items.
            Defaults to None.

        Returns
        -------
        None
        """
        self.name = name
        self.num_items = num_items
        self.ts_type = ts_type
        self.data_type = data_type
        self.interpolation = interpolation
        self.item_format = item_format
        self.item_info = item_info

    @classmethod
    def from_xml(cls, source):
        """
        Create a DataSource instance from XML.

        Parameters
        ----------
        source : str or ElementTree.Element or bytes or bytearray or file-like object
            The XML source to parse and create a DataSource instance.

        Returns
        -------
        DataSource
            An instance of DataSource created from the XML source.

        Raises
        ------
        ValueError
            If the XML source type is not supported or if the XML structure is invalid.

        Notes
        -----
        This class method reads an object from XML, extracting information to create
        a DataSource instance.

        Examples
        --------
        >>> xml_string = "<DataSource Name='Example' NumItems='2'><TSType>...</TSType></DataSource>"
        >>> data_source_instance = DataSource.from_xml(xml_string)
        >>> isinstance(data_source_instance, DataSource)
        True
        """
        if isinstance(source, str):
            # If the source is a string, treat it as raw XML
            root = DefusedElementTree.fromstring(source)
        elif isinstance(source, ElementTree.Element):
            # If the source is an ElementTree object, use it directly
            root = source
        elif isinstance(source, (bytes, bytearray)):
            # If the source is a bytes or bytearray, assume it's
            # XML content and decode it.
            root = DefusedElementTree.fromstring(source.decode())
        elif hasattr(source, "read"):
            # If the source has a 'read' method, treat it as a
            # file-like object.
            root = DefusedElementTree.parse(source).getroot()
        else:
            raise ValueError("Unsupported XML source type.")

        if root.tag == "DataSource":
            name = root.attrib["Name"]
            num_items = int(root.attrib["NumItems"])
        else:
            raise ValueError(
                "Tag at DataSource level should be 'DataSource'," f" found {root.tag}."
            )

        ts_type = root.findtext("TSType")
        data_type = root.findtext("DataType")
        interpolation = root.findtext("Interpolation")
        item_format = root.findtext("ItemFormat")

        item_infos_raw = root.findall("ItemInfo")
        if (len(item_infos_raw) != num_items) and (num_items > 1):
            raise ValueError(
                f"Malformed Hilltop XML. DataSource {name} expects {num_items} "
                f"ItemInfo(s), but found {len(item_infos_raw)}"
            )

        item_info_list = []
        for info in item_infos_raw:
            item_info_list += [ItemInfo.from_xml(info)]

        return cls(
            name,
            num_items,
            str(ts_type),
            str(data_type),
            str(interpolation),
            str(item_format),
            item_info_list,
        )

    def to_xml_tree(self):
        """
        Convert the DataSource instance to an XML ElementTree.

        Returns
        -------
        ElementTree.Element
            The XML ElementTree representing the DataSource instance.

        Notes
        -----
        This method converts the DataSource object into an XML blob
        using ElementTree.

        Examples
        --------
        >>> name = "Example"
        >>> num_items = 2
        >>> ts_type = "..."
        >>> data_type = "..."
        >>> interpolation = "..."
        >>> item_format = "..."
        >>> item_info = [ItemInfo(...), ItemInfo(...)]  # Replace '...' with appropriate arguments
        >>> data_source_instance = DataSource(name, num_items, ts_type, data_type, interpolation, item_format, item_info)
        >>> xml_tree = data_source_instance.to_xml_tree()
        >>> isinstance(xml_tree, ElementTree.Element)
        True
        """
        data_source_root = ElementTree.Element(
            "DataSource",
            attrib={"Name": self.name, "NumItems": str(self.num_items)},
        )

        ts_type_element = ElementTree.SubElement(data_source_root, "TSType")
        ts_type_element.text = self.ts_type

        data_type_element = ElementTree.SubElement(data_source_root, "DataType")
        data_type_element.text = self.data_type

        interpolation_element = ElementTree.SubElement(
            data_source_root, "Interpolation"
        )
        interpolation_element.text = self.interpolation

        item_format_element = ElementTree.SubElement(data_source_root, "ItemFormat")
        item_format_element.text = self.item_format

        if self.item_info is not None:
            data_source_root.extend(
                [element.to_xml_tree() for element in self.item_info]
            )
        return data_source_root


class Data:
    """Data Class."""

    def __init__(
        self,
        date_format: str,
        num_items: int,
        timeseries: pd.Series | pd.DataFrame,
    ):
        """
        Initialize a Data instance.

        Parameters
        ----------
        date_format : str
            The date format associated with the data.
        num_items : int
            The number of items in the data.
        timeseries : pd.Series or pd.DataFrame
            The timeseries associated with the data. For a single-item data, a pd.Series
            is expected, and for multi-item data, a pd.DataFrame is expected.

        Returns
        -------
        None
        """
        self.date_format = date_format
        self.num_items = num_items
        self.timeseries = timeseries

    @classmethod
    def from_xml(cls, source):
        """
        Create a Data instance from XML.

        Parameters
        ----------
        source : str or ElementTree.Element or bytes or bytearray or file-like object
            The XML source to parse and create a Data instance.

        Returns
        -------
        Data
            An instance of Data created from the XML source.

        Raises
        ------
        ValueError
            If the XML source type is not supported or if the XML structure is invalid.

        Notes
        -----
        This class method reads an object from XML, extracting information to create
        a Data instance.

        Examples
        --------
        >>> xml_string = "<Data DateFormat='%Y-%m-%d' NumItems='1'><T>2023-01-01</T><V>42.0</V></Data>"
        >>> data_instance = Data.from_xml(xml_string)
        >>> isinstance(data_instance, Data)
        True
        """
        if isinstance(source, str):
            # If the source is a string, treat it as raw XML
            root = DefusedElementTree.fromstring(source)
        elif isinstance(source, ElementTree.Element):
            # If the source is an ElementTree object, use it directly
            root = source
        elif isinstance(source, (bytes, bytearray)):
            # If the source is a bytes or bytearray, assume it's
            # XML content and decode it.
            root = DefusedElementTree.fromstring(source.decode())
        elif hasattr(source, "read"):
            # If the source has a 'read' method, treat it as a
            # file-like object.
            root = DefusedElementTree.parse(source).getroot()
        else:
            raise ValueError("Unsupported XML source type.")

        if root.tag == "Data":
            date_format = root.attrib["DateFormat"]
            num_items = int(root.attrib["NumItems"])
        else:
            raise ValueError(
                "Tag at Data level should be 'Data'," f" found {root.tag}."
            )

        data_list = []
        for child in root:
            if num_items > 1:
                if child.tag != "E":
                    # Multivariate data seems to always be tagged with E
                    raise ValueError("Malformed Hilltop XML.")
                data_dict = {}
                for element in child:
                    data_dict[element.tag] = element.text

                data_list += [data_dict]
            else:
                if child.text is not None:
                    timestamp, data_val = child.text.split(" ")
                    data_dict = {
                        "T": timestamp,
                        "V": data_val,
                    }
                    data_list += [data_dict]

        if num_items > 1:
            timeseries = pd.DataFrame(data_list).set_index("T")
        else:
            timeseries = pd.Series(
                [dat["V"] for dat in data_list],
                index=[dat["T"] for dat in data_list],
            )

        return cls(date_format, num_items, timeseries)

    def to_xml_tree(self):
        """
        Convert the Data instance to an XML ElementTree.

        Returns
        -------
        ElementTree.Element
            The XML ElementTree representing the Data instance.

        Notes
        -----
        This method converts the Data object into an XML blob
        using ElementTree.

        Examples
        --------
        >>> date_format = "%Y-%m-%d"
        >>> num_items = 1
        >>> timeseries = pd.Series([42.0], index=["2023-01-01"])
        >>> data_instance = Data(date_format, num_items, timeseries)
        >>> xml_tree = data_instance.to_xml_tree()
        >>> isinstance(xml_tree, ElementTree.Element)
        True
        """
        data_root = ElementTree.Element(
            "Data",
            attrib={
                "DateFormat": self.date_format,
                "NumItems": str(self.num_items),
            },
        )

        if self.num_items == 1:
            if isinstance(self.timeseries, pd.Series):
                for idx, val in self.timeseries.items():
                    element = ElementTree.SubElement(data_root, "V")
                    element.text = f"{idx} {val}"
            else:
                raise TypeError(
                    "pandas Series expected for data with single field."
                    f" Found {type(self.timeseries)}."
                )
        else:
            if isinstance(self.timeseries, pd.DataFrame):
                for date, row in self.timeseries.iterrows():
                    element = ElementTree.SubElement(data_root, "E")
                    timestamp = ElementTree.SubElement(element, "T")
                    timestamp.text = str(date)
                    for i, val in enumerate(row):
                        val_item = ElementTree.SubElement(element, f"I{i+1}")
                        val_item.text = str(val)
            else:
                raise TypeError(
                    "pandas DataFrame expected for data with multiple fields."
                    f" Found {type(self.timeseries)}."
                )
        return data_root


class DataSourceBlob:
    """DataSourceBlob class."""

    def __init__(
        self,
        site_name: str,
        data_source: DataSource,
        data: Data,
        tideda_site_number: str | None = None,
    ):
        """
        Initialize a DataSourceBlob instance.

        Parameters
        ----------
        site_name : str
            The name of the site associated with the data.
        data_source : DataSource
            The DataSource object containing information about the data source.
        data : Data
            The Data object containing the actual data.
        tideda_site_number : str or None, optional
            The Tideda site number, if available. Defaults to None.

        Returns
        -------
        None
        """
        self.site_name = site_name
        self.data_source = data_source
        self.data = data
        self.tideda_site_number = tideda_site_number

    @classmethod
    def from_xml(cls, source):
        """
        Create a DataSourceBlob instance from XML.

        Parameters
        ----------
        source : str or ElementTree.Element or bytes or bytearray or file-like object
            The XML source to parse and create a DataSourceBlob instance.

        Returns
        -------
        DataSourceBlob
            An instance of DataSourceBlob created from the XML source.

        Raises
        ------
        ValueError
            If the XML source type is not supported or if the XML structure is invalid.

        Notes
        -----
        This class method reads an object from XML, extracting information to create
        a DataSourceBlob instance.

        Examples
        --------
        >>> xml_string = "<Measurement SiteName='Example'><Data>...</Data></Measurement>"
        >>> data_source_blob = DataSourceBlob.from_xml(xml_string)
        >>> isinstance(data_source_blob, DataSourceBlob)
        True
        """
        if isinstance(source, str):
            # If the source is a string, treat it as raw XML
            root = DefusedElementTree.fromstring(source)
        elif isinstance(source, ElementTree.Element):
            # If the source is an ElementTree object, use it directly
            root = source
        elif isinstance(source, (bytes, bytearray)):
            # If the source is a bytes or bytearray, assume it's
            # XML content and decode it.
            root = DefusedElementTree.fromstring(source.decode())
        elif hasattr(source, "read"):
            # If the source has a 'read' method, treat it as a
            # file-like object.
            root = DefusedElementTree.parse(source).getroot()
        else:
            raise ValueError("Unsupported XML source type.")

        if root.tag == "Measurement":
            site_name = root.attrib["SiteName"]
        else:
            raise ValueError(
                "Tag at Measurement level should be 'Measurement',"
                f" found {root.tag}."
            )

        tideda_site_number = root.findtext("TidedaSiteNumber")

        data_source_element = root.find("DataSource")
        data_source = DataSource.from_xml(data_source_element)

        data_element = root.find("Data")
        data = Data.from_xml(data_element)

        return cls(site_name, data_source, data, tideda_site_number)

    def to_xml_tree(self):
        """
        Convert the DataSourceBlob instance to an XML ElementTree.

        Returns
        -------
        ElementTree.Element
            The XML ElementTree representing the DataSourceBlob instance.

        Notes
        -----
        This method converts the DataSourceBlob object into an XML blob
        using ElementTree.

        Examples
        --------
        >>> data_source_blob = DataSourceBlob("Example", data_source, data, "123")
        >>> xml_tree = data_source_blob.to_xml_tree()
        >>> isinstance(xml_tree, ElementTree.Element)
        True
        """
        data_source_blob_root = ElementTree.Element(
            "Measurement", attrib={"SiteName": self.site_name}
        )
        data_source_blob_root.append(self.data_source.to_xml_tree())
        if self.tideda_site_number is not None:
            tideda_site_number_element = ElementTree.SubElement(
                data_source_blob_root, "TidedaSiteNumber"
            )
            tideda_site_number_element.text = str(self.tideda_site_number)
        data_source_blob_root.append(self.data.to_xml_tree())

        return data_source_blob_root


def parse_xml(source) -> list[DataSourceBlob]:
    """
    Parse Hilltop XML to get a list of DataSourceBlob objects.

    Parameters
    ----------
    source : str or ElementTree.Element or bytes or bytearray or file-like object
        The source XML to parse. It can be a raw XML string, an ElementTree object,
        XML content in bytes or bytearray, or a file-like object.

    Returns
    -------
    List[DataSourceBlob]
        A list of DataSourceBlob objects parsed from the Hilltop XML.

    Raises
    ------
    ValueError
        If the source type is not supported or if the XML structure is possibly malformed.

    Notes
    -----
    This function parses Hilltop XML and extracts information to create DataSourceBlob objects.
    The DataSourceBlob objects contain data from Measurement elements in the Hilltop XML.

    Examples
    --------
    >>> xml_string = "<Hilltop><Measurement>...</Measurement></Hilltop>"
    >>> data_source_blobs = parse_xml(xml_string)
    >>> len(data_source_blobs)
    1
    >>> isinstance(data_source_blobs[0], DataSourceBlob)
    True
    """
    if isinstance(source, str):
        # If the source is a string, treat it as raw XML
        root = DefusedElementTree.fromstring(source)
    elif isinstance(source, ElementTree.Element):
        # If the source is an ElementTree object, use it directly
        root = source
    elif isinstance(source, (bytes, bytearray)):
        # If the source is a bytes or bytearray, assume it's
        # XML content and decode it.
        root = DefusedElementTree.fromstring(source.decode())
    elif hasattr(source, "read"):
        # If the source has a 'read' method, treat it as a
        # file-like object.
        root = DefusedElementTree.parse(source).getroot()
    else:
        raise ValueError("Unsupported XML source type.")

    if root.tag != "Hilltop":
        raise ValueError(
            f"Possibly malformed Hilltop xml. Root tag is '{root.tag}',"
            " should be 'Hilltop'."
        )
    data_source_blob_list = []
    for child in root.iter():
        if child.tag == "Measurement":
            data_source_blob = DataSourceBlob.from_xml(child)
            if data_source_blob.data_source.item_info is not None:
                item_names = [
                    info.item_name for info in data_source_blob.data_source.item_info
                ]
            else:
                item_names = []
            if len(item_names) > 1 and isinstance(
                data_source_blob.data.timeseries, pd.DataFrame
            ):
                cols = {
                    col: name
                    for col, name in zip(
                        data_source_blob.data.timeseries.columns,
                        item_names,
                    )
                }
                data_source_blob.data.timeseries = (
                    data_source_blob.data.timeseries.rename(columns=cols)
                )
            elif len(item_names) > 0:
                data_source_blob.data.timeseries.name = item_names[0]
            else:
                # It seems that if iteminfo is missing it's always a QC
                data_source_blob.data.timeseries.name = "QualityCode"
            data_source_blob.data.timeseries.index.name = "Time"
            data_source_blob_list += [data_source_blob]

    return data_source_blob_list


def write_hilltop_xml(data_source_blob_list, output_path):
    """
    Write Hilltop XML file based on a list of DataSourceBlob objects.

    Parameters
    ----------
    data_source_blob_list : list[DataSourceBlob]
        List of DataSourceBlob objects to be included in the Hilltop XML file.
    output_path : str
        The path to the output XML file. If the file already exists, it will be overwritten.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the data_source_blob_list is not a list of DataSourceBlob objects.

    Notes
    -----
    This function takes a list of DataSourceBlob objects and writes a Hilltop XML file
    using the ElementTree module. The resulting XML file follows the Hilltop schema.

    The XML file structure includes an 'Agency' element with the text content set to "Horizons",
    and then a series of elements generated from the DataSourceBlob objects in the provided list.
    The XML file is encoded in UTF-8, and it includes an XML declaration at the beginning.

    Examples
    --------
    >>> blob_list = [dataSourceBlob1, dataSourceBlob2, dataSourceBlob3]
    >>> write_hilltop_xml(blob_list, "output.xml")

    The above example writes a Hilltop XML file named "output.xml" based on the provided
    list of DataSourceBlob objects.

    """
    root = ElementTree.Element("Hilltop")
    agency = ElementTree.Element("Agency")
    agency.text = "Horizons"
    root.append(agency)

    for blob in data_source_blob_list:
        elem = blob.to_xml_tree()
        root.append(elem)

    ElementTree.indent(root)
    etree = ElementTree.ElementTree(element=root)

    # Write the XML file to the specified output path
    etree.write(output_path, encoding="utf-8", xml_declaration=True)
