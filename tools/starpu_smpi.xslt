<!--
 StarPU : Runtime system for heterogeneous multicore architectures.

 Copyright (C) 2014-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria

 StarPU is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.1 of the License, or (at
 your option) any later version.

 StarPU is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 See the GNU Lesser General Public License in COPYING.LGPL for more details.
-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

	<xsl:output doctype-system="http://simgrid.gforge.inria.fr/simgrid.dtd"/>

     <!-- Add doctype 
     <xsl:text>&lt;!DOCTYPE platform SYSTEM 'http://simgrid.gforge.inria.fr/simgrid.dtd'&gt;</xsl:text>

-->
    <!-- Copy everything by default but keep applying templates.  -->
    <xsl:template match="platform|AS|host|link|prop|route|link_ctn|@*">
        <xsl:copy>
            <xsl:apply-templates select="node()|@*"/>
        </xsl:copy>
    </xsl:template>

    <!-- Replace AS name.  -->
    <xsl:template match="platform/AS/@id">
        <xsl:attribute name="id">
            <xsl:value-of select="$ASname"/>
        </xsl:attribute>
    </xsl:template>

    <!-- Prepend AS name to host names.  -->
    <xsl:template match="platform/AS/host/@id">
	    <xsl:attribute name="id"><xsl:value-of select="$ASname"/>-<xsl:value-of select="."/></xsl:attribute>
    </xsl:template>
    <xsl:template match="platform/AS/link/@id">
	    <xsl:attribute name="id"><xsl:value-of select="$ASname"/>-<xsl:value-of select="."/></xsl:attribute>
    </xsl:template>
    <xsl:template match="platform/AS/route/@src">
	    <xsl:attribute name="src"><xsl:value-of select="$ASname"/>-<xsl:value-of select="."/></xsl:attribute>
    </xsl:template>
    <xsl:template match="platform/AS/route/@dst">
	    <xsl:attribute name="dst"><xsl:value-of select="$ASname"/>-<xsl:value-of select="."/></xsl:attribute>
    </xsl:template>
    <xsl:template match="platform/AS/route/link_ctn/@id">
	    <xsl:attribute name="id"><xsl:value-of select="$ASname"/>-<xsl:value-of select="."/></xsl:attribute>
    </xsl:template>

</xsl:stylesheet>


