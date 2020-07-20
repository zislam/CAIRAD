/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 * Copyright (C) 2009 Amri Napolitano
 */
package weka.filters.unsupervised.attribute;

import weka.core.Instances;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;

import junit.framework.Test;
import junit.framework.TestSuite;

/**
 * Tests CAIRADTest. Run from the command line with:
 * <p>
 * java weka.filters.unsupervised.attribute.CAIRADTest
 *
 * @author Amri Napolitano
 * @version $Revision: 8108 $
 */
public class CAIRADTest extends AbstractFilterTest {

    public CAIRADTest(String name) {
        super(name);
        this.m_FilteredClassifier = null;
    }

    /**
     * Creates a default CAIRAD
     */
    @Override
    public Filter getFilter() {
        this.m_FilteredClassifier = null;
        return new CAIRAD();
    }

    public void testTypical() {
        this.m_FilteredClassifier = null;
        Instances result = useFilter();
        // Number of attributes and instances shouldn't change
        assertEquals(m_Instances.numAttributes(), result.numAttributes());
        assertEquals(m_Instances.numInstances(), result.numInstances());
    }

    @Override
    public void testIncremental() {
        Instances icopy = new Instances(m_Instances);

        try {
            m_Filter.setInputFormat(icopy);
            for (int i = 0; i < icopy.numInstances(); i++) {
                m_Filter.input(icopy.get(i));
            }

            m_Filter.batchFinished();
            Instances result = m_Filter.getOutputFormat();
            while (m_Filter.numPendingOutput() > 0) {
                result.add(m_Filter.output());
            }
            System.out.println(result.toString());
        } catch (Exception e) {
            e.printStackTrace();
            fail("Incremental failed: " + e.toString());
        }

    }

    public static Test suite() {
        return new TestSuite(CAIRADTest.class);
    }

    public static void main(String[] args) {
        junit.textui.TestRunner.run(suite());
    }
}
