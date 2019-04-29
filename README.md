
# Multi-Agent Actor-Critic Learning using CUDA to solve mine game. 

There are 512 agents with dimension 46 by 46. The idea is to take advantage of the paralellism properties of Cuda to optimize and make the agents solve the maze as fast as they can.  
### Technical Approach

Each thread associates with one agent, therefore in total we will have 512 threads. However, all agent update the same Q table later will be use as critic. Every time agent steps on mine, the enviroment will return -1, flag + 1 or 0 else. The game continues until convergence.



### Result
![Screenshot](Figure_1.png)
![Screenshot](Figure_3.png)
## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Adrian Mai* 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

