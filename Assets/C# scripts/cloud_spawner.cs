using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class cloud_spawner : MonoBehaviour
{
    public GameObject cloud_group_prefab;
    public float spawnRate = 10f;
    private float timer = 0f;

    // Start is called before the first frame update
    void Start()
    {
        spawnGroupOfClouds();
    }

    // Update is called once per frame
    void Update()
    {
        // Check if the application is playing (not in the Unity Editor)
        if (!Application.isEditor || Application.isPlaying)
        {
            if (timer < spawnRate)
            {
                timer = timer + Time.deltaTime;
            }
            else
            {
                spawnGroupOfClouds();
                timer = 0;
            }
        }
    }


    void spawnGroupOfClouds()
    {

        GameObject group_of_clouds_clone = Instantiate(cloud_group_prefab, transform.position, transform.rotation);

        // To give the group of clouds clone the 'cloud_movement' C# script component
        cloud_movement_and_destruction cloud_movement_script_reference = group_of_clouds_clone.GetComponent<cloud_movement_and_destruction>();

        if (cloud_movement_script_reference == null)
        {
            cloud_movement_script_reference = group_of_clouds_clone.AddComponent<cloud_movement_and_destruction>();
        }
    }
}

