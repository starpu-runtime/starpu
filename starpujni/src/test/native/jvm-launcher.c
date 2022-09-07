//
// Created by Gérald Point on 07/02/2019.
//

#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int s_start_jvm (JavaVM **p_jvm, JNIEnv **p_env, const char *jarfile);
static int s_terminate_jvm (JavaVM *jvm);
static int s_invoke_main_from_class (JavaVM *jvm, JNIEnv *env, const char *classname, int argc, char **argv);

int main (int argc, char **argv)
{
	int ret = EXIT_SUCCESS;
	JavaVM *jvm;
	JNIEnv *env;

	printf ("# JVM-LAUNCHER\ninput jar: %s\nstartup class: %s\nargs : ",
		STARPUJNI_JARFILE,
		STARPUJNI_LAUNCHER_CLASS);
	for (int i = 1; i < argc; i++)
	{
		printf ("%s ", argv[i]);
	}
	printf ("\n");

	if (s_start_jvm (&jvm, &env, STARPUJNI_JARFILE))
	{
		if (!s_invoke_main_from_class (jvm, env, STARPUJNI_LAUNCHER_CLASS,
					       argc - 1, argv + 1))
		{
			(*env)->ExceptionDescribe (env);
			ret = EXIT_FAILURE;
		}
		if (!s_terminate_jvm (jvm))
			ret = EXIT_FAILURE;
	}
	return ret;
}

static jobjectArray s_argv_to_array (JNIEnv *env, int argc, char **argv)
{
	jclass stringClass = (*env)->FindClass (env, "java/lang/String");
	jobjectArray result = (*env)->NewObjectArray (env, argc, stringClass, NULL);

	for (int i = 0; i < argc; i++)
	{
		jstring str = (*env)->NewStringUTF (env, argv[i]);
		(*env)->SetObjectArrayElement (env, result, i, str);
	}

	return result;
}

static int s_invoke_main_from_class (JavaVM *jvm, JNIEnv *env, const char *classname, int argc, char **argv)
{
	int result = 0;
	jclass theClass = (*env)->FindClass (env, classname);

	if (theClass == NULL)
		fprintf (stderr, "can't find class '%s'.\n", classname);
	else
	{
		jmethodID mID = (*env)->GetStaticMethodID (env, theClass, "main",
							   "([Ljava/lang/String;)V");
		if (mID != NULL)
		{
			jobjectArray args = s_argv_to_array (env, argc, argv);
			(*env)->CallStaticVoidMethod (env, theClass, mID, args);
			result = ((*env)->ExceptionCheck (env) == JNI_FALSE);
		}
	}
	return result;
}

static char *s_build_classpath (const char *jarfile)
{
	static const char *jcpopt = "-Djava.class.path=";
	char *result = NULL;
	char *envcp = getenv ("CLASSPATH");
	size_t result_size = strlen (jarfile) + strlen (jcpopt) + 1;

	if (envcp == NULL)
	{
		fprintf (stderr, "CLASSPATH is not set. The program may not find classes"
			 "related to external tools like Hadoop.\n");
	}
	else
	{
		result_size += strlen (envcp) + 1;
	}
	result = calloc (result_size, sizeof (char));
	strcat (result, jcpopt);
	strcat (result, jarfile);
	if (envcp != NULL)
	{
		strcat (result, ":");
		strcat (result, envcp);
	}

	return result;
}

static int s_start_jvm (JavaVM **p_jvm, JNIEnv **p_env, const char *jarfile)
{
	JavaVMInitArgs vm_args;
	JavaVMOption options[1];
	int result = 0;

	options[0].optionString = s_build_classpath (jarfile);
	//options[1].optionString = "-verbose:class";
	//options[2].optionString = "-verbose:jni";

	vm_args.version = JNI_VERSION_1_8;
	vm_args.nOptions = sizeof (options) / sizeof (options[0]);
	vm_args.options = options;
	vm_args.ignoreUnrecognized = JNI_FALSE;

	/* load and initialize a Java VM, return a JNI interface pointer in ctx */
	if (JNI_CreateJavaVM (p_jvm, (void **) p_env, &vm_args) == JNI_OK)
		result = 1;
	free (options[0].optionString);
	return result;
}

static int s_terminate_jvm (JavaVM *jvm)
{
	return ((*jvm)->DestroyJavaVM (jvm) == JNI_OK);
}
